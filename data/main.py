import os
import tasks_asrnew
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch
import csv


input_path = 'NLP\\audio_transfer\\audio'
dir_files = os.walk(input_path)
audio_files = []
audio_name = []

for root, dirs, files in dir_files:
    for file in files:
        if any(file.endswith(ext) for ext in ['.mp3','.wav']):
                audio_files.append(os.path.join(root, file))
                audio_name.append(os.path.splitext(file)[0])
    
sent_output_path = 'NLP\\audio_transfer\\segmented_audio\\sent_audio'
dialog_output_path = 'NLP\\audio_transfer\\segmented_audio\\dialog_audio'
sent_path = 'NLP\\audio_transfer\\segmented_audio\\label\\sentence_test.csv'
dialog_path = 'NLP\\audio_transfer\\segmented_audio\\label\\dialog_test.csv'

sent_file = open(sent_path, mode='w', newline='', encoding='utf-8')
sent_writer = csv.writer(sent_file)
sent_writer.writerow(['Original_File', 'Audio_id', 'Segment_number', 'Text', 'Lang', 'Confidence'])

dialog_file = open(dialog_path, mode='w', newline='', encoding='utf-8')
dialog_writer = csv.writer(dialog_file)
dialog_writer.writerow(['Original_File', 'Audio_id', 'Segment_number', 'Text', 'Lang', 'Confidence'])


model_size = "large-v3"
asr_model = WhisperModel(model_size, device="cuda", compute_type="float16")
spk_rec_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
spk_rec_pipeline.to(torch.device("cuda"))


count = 0


for index, file_name in enumerate(audio_files[1:100]):
    print('processing {}: '.format(audio_name[index]), end = '')
    wav_file_name = f"{os.path.splitext(file_name)[0]}.wav"
    audio = AudioSegment.from_mp3(file_name)
    audio.export(wav_file_name, format="wav")            # 会保留原来的通道数，除非进行了操作
    segment_filepath = wav_file_name
    audio = AudioSegment.from_wav(segment_filepath)
    task_result, dialog_result = tasks_asrnew.process_asr_task(segment_filepath, asr_model, spk_rec_pipeline)

    print('exporting segmented files and label', end='--')
    for task_dict in task_result:
        id, start, end, speaker, text, confid = task_dict.values()
        start_ms, end_ms = start * 1000, end * 1000
        segment_audio = audio[start_ms : end_ms]
        segment_audio.export("{}\\{}-Segment{}.wav".format(sent_output_path, audio_name[index], id))
        sent_writer.writerow([
            audio_name[index],
            "{}-Segment{}.wav".format(audio_name[index], id),
            id,
            speaker + ': ' + text,
            'zh',
            confid
            ])

    for dialog in dialog_result:
        id, start, end, text, confid = dialog.values()
        start_ms, end_ms = start * 1000, end * 1000
        segment_audio = audio[start_ms : end_ms]
        segment_audio.export("{}\\{}-Dialog{}.wav".format(dialog_output_path, audio_name[index], id))
        dialog_writer.writerow([
            audio_name[index], 
            "{}-Dialog{}.wav".format(audio_name[index], id),
            id,
            text,
            'zh',
            confid
            ])
    if len(task_result) == 1:
        count += 1
    os.remove(segment_filepath)
    print(count)

dialog_file.close()
sent_file.close()