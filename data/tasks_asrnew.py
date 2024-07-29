from celery import shared_task
import pandas as pd
import os
from copy import deepcopy
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.core import Segment
import torch
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 初始化 WhisperModel（在任务中创建实例或在模块级别创建实例取决于你的需求）
# model_size = "large-v3"
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

@shared_task
def process_asr_task(segment_filepath, asr_model, spk_rec_pipeline):
    # 使用 fastwhisper 处理音频
    # model_size = "large-v3"
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # segments, info = model.transcribe(segment_filepath, beam_size=5, language="zh")
    # test_segments = list(segment.text for segment in segments)
    # # transcription = " ".join([segment.text for segment in segments])
    # transcription = " ".join(test_segments)



    def get_text_with_timestamp(transcribe_res):          
        timestamp_texts = []
        for item in transcribe_res:
            start = item.start
            end = item.end
            text = item.text.strip()
            confidence = 1 - float(item.no_speech_prob)
            timestamp_texts.append((Segment(start, end), text, confidence))
        return timestamp_texts

    def add_speaker_info_to_text(timestamp_texts, ann): 
        spk_text = []
        for seg, text, confid in timestamp_texts:
            spk = ann.crop(seg).argmax()
            spk_text.append((seg, spk, text,confid))
        return spk_text
    
    def get_saler(spk_text, all_speakers):
        certain_key_words = ['我们是', '我是', '我们这边是', '我这边是', '请问']
        potential_key_words = ['项目', '活动', '了解', '以前', '上次']
        saler = None
        # 查看前两句，计数
        for seg, spk, text, confid in spk_text[:2]:
            count = 0
            for word in certain_key_words:
                if word in text:
                    count += 1
            for word in potential_key_words:
                if word in text:
                    count += 1
            if count >= 2:
                saler = spk

        # 根据前两句的出现说话者是否相同，以及对话长度
        if saler is None:
            spk_1, first_text_len = spk_text[0][1], len(spk_text[0][2])
            spk_2, second_text_len = spk_text[0][1], len(spk_text[0][2])
            if spk_1 == spk_2:
                saler = spk_1
            else:
                saler = spk_1 if first_text_len > second_text_len else spk_2
                
        # 得到另一方的spk
        all_speakers.remove(saler)
        sc_dict = { key : "顾客" for key in all_speakers}
        sc_dict[saler] = '客服'
        return sc_dict

    def merge_cache(text_cache):
        sentence = ''.join([item[2] for item in text_cache])
        spk = text_cache[0][1]
        start = round(text_cache[0][0].start, 1)
        end = round(text_cache[-1][0].end, 1)
        confid = [sent[3] for sent in text_cache]
        return Segment(start, end), spk, sentence, min(confid)

    PUNC_SENT_END = [',', '.', '?', '!', "，", "。", "？", "！"]

    def merge_sentence(spk_text, sc_dict):
        merged_spk_text = []
        pre_spk = None
        text_cache = []
        for seg, spk, text, confid in spk_text:
            if spk == None:
                continue
            if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:    # 表示更换了spk
                merged_spk_text.append(merge_cache(text_cache))
                text_cache = [(seg, sc_dict[spk], text, confid)]
                pre_spk = spk

            elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
                text_cache.append((seg, sc_dict[spk], text, confid))
                pre_spk = spk
            else:
                text_cache.append((seg, sc_dict[spk], text, confid))
                pre_spk = spk
        if len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
        return merged_spk_text
    
    def merge_dialog(sentence_result):
        dialog = []
        for i in range(1, len(sentence_result), 2):
            first_sentence = sentence_result[i - 1]
            second_sentence = sentence_result[i]
            start = first_sentence[0].start
            end = second_sentence[0].end
            text = "{}: {}; {}: {}".format(first_sentence[1], first_sentence[2], second_sentence[1], second_sentence[2])
            confid = min(first_sentence[3], second_sentence[3])
            dialog.append((Segment(start, end), text, confid))
        if len(sentence_result) % 2 == 1:
            dialog.append((sentence_result[-1][0], sentence_result[-1][1] + ": " + sentence_result[-1][2], sentence_result[-1][3]))
        return dialog

    def diarize_text(transcribe_res, diarization_result):
        print("segmenting audio", end= '-')
        timestamp_texts = get_text_with_timestamp(transcribe_res)
        print("classify speakers", end= '-')
        spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
        print("distinguish saler and customer", end= '-')
        saler_customer_dict = get_saler(spk_text, diarization_result.labels())
        print("merge consecutive audio", end= '-')
        sentence_result = merge_sentence(spk_text, saler_customer_dict)
        print("merge dialog", end='-')
        dialog_result = merge_dialog(sentence_result)
        return sentence_result, dialog_result

    print('start transforming', end= '-')
    asr_result, info = asr_model.transcribe(segment_filepath, beam_size=5, language="zh")
    diarization_result = spk_rec_pipeline(segment_filepath)
    sentence_result, dialog_result = diarize_text(asr_result, diarization_result)
    

    task_result = []
    count = 0
    for segment, spk, sent, confid in sentence_result:
        count+=1
        task_result.append({
            "id": count,
            "start": segment.start,
            "end": segment.end,
            "speaker": spk,
            "text": sent,
            "confid" : confid
        })

    count = 0
    dialog_result_id = []
    for segment, sent, confid in dialog_result:
        count+=1
        dialog_result_id.append({
            "id": count,
            "start": segment.start,
            "end": segment.end,
            "text": sent,
            "confid": confid
        })

    return task_result, dialog_result_id



