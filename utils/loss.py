def train(model, train_loader, criterion, optimizer, device, epoch: int, lambda_L1=0.000001, lambda_L2=0.0000001):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for texts, audios, videos, labels, masks in train_loader:
        texts, audios, videos, labels, masks = (d.to(device) for d in (texts, audios, videos, labels, masks))
        optimizer.zero_grad()
        outputs = model(texts, audios, videos, masks)
        labels = labels.float()
        
        # Regular loss calculation
        loss = criterion(outputs, labels)

        # L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        # L2 regularization
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        
        ent_loss = entropy_loss(outputs)
        neg_ent_loss = -ent_loss
        for name, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                mma_loss = get_mma_loss(m.weight)
                loss = loss + 0.00007 * mma_loss
        # if epoch < 60:
        #     loss = loss + lambda_L1 * l1_norm + lambda_L2 * l2_norm + 0.01 * ent_loss
        # else:
        #     loss = loss + lambda_L1 * l1_norm + lambda_L2 * l2_norm +  10 * ent_loss
        loss = loss + lambda_L1 * l1_norm + lambda_L2 * l2_norm
        if epoch > 60:
            loss +=  0.1*neg_ent_loss
        loss.backward()
        optimizer.step()

        masks_expanded = masks.unsqueeze(-1).expand_as(labels)
        total_loss += loss.item()
        pred = torch.sigmoid(outputs) > 0.5
        correct += ((pred == labels.byte()) * masks_expanded.byte()).sum().item()
        total += masks_expanded.sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', accuracy, epoch)
    return avg_loss, accuracy