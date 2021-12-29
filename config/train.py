MOMENTUM = os.environ(['MOMENTUM'])
LEARNING_RATE = os.environ(['LEARNING_RATE'])
WEIGHT_DECAY = os.environ(['WEIGHT_DECAY'])

if (__name__ == '__main__') or (__name__ == 'config.train'):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    #optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    n_batches, n_batches_val = len(train_dataset), len(test_dataset)

    validation_mask_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Starting epoch {epoch} of {NUM_EPOCHS}")

        time_start = time.time()
        loss_accum = 0.0
        loss_mask_accum = 0.0
        loss_classifier_accum = 0.0
        # for batch_idx, (images, targets) in enumerate(dl_train, 1):
        for batch_idx, (images, targets) in enumerate(dl_train):
        
            # Predict
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            print(type(loss_dict), len(loss_dict))
            loss = sum(loss for loss in loss_dict.values())
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            loss_mask = loss_dict['loss_mask'].item()
            loss_accum += loss.item()
            loss_mask_accum += loss_mask
            loss_classifier_accum += loss_dict['loss_classifier'].item()
            
            if batch_idx % 400 == 0:
                print(f"    [Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_mask:7.3f}.")
                            
        if USE_SCHEDULER:
            lr_scheduler.step()

        # Train losses
        train_loss = loss_accum / n_batches
        train_loss_mask = loss_mask_accum / n_batches
        train_loss_classifier = loss_classifier_accum / n_batches

        # Validation
        val_loss_accum = 0
        val_loss_mask_accum = 0
        val_loss_classifier_accum = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dl_val, 1):
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                val_loss_dict = model(images, targets)
                val_batch_loss = sum(loss for loss in val_loss_dict.values())
                val_loss_accum += val_batch_loss.item()
                val_loss_mask_accum += val_loss_dict['loss_mask'].item()
                val_loss_classifier_accum += val_loss_dict['loss_classifier'].item()

                # save_general_checkpoint(model, optimizer, NUM_EPOCHS, f"iter_{BATCH_SIZE}", "seg", batch_idx, val_loss_accum)

        # Validation losses
        val_loss = val_loss_accum / n_batches_val
        val_loss_mask = val_loss_mask_accum / n_batches_val
        val_loss_classifier = val_loss_classifier_accum / n_batches_val
        elapsed = time.time() - time_start

        validation_mask_losses.append(val_loss_mask)

        torch.save(model.state_dict(), f"pytorch_model-e{epoch}.pth")
        prefix = f"[Epoch {epoch:2d} / {NUM_EPOCHS:2d}]"
        # print(prefix)
        print(f"{prefix} Train mask-only loss: {train_loss_mask:7.3f}, classifier loss {train_loss_classifier:7.3f}")
        print(f"{prefix} Val mask-only loss  : {val_loss_mask:7.3f}, classifier loss {val_loss_classifier:7.3f}")
        print(prefix)
        print(f"{prefix} Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f} [{elapsed:.0f} secs]")
        # print(prefix)