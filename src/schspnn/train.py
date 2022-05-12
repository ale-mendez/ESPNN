def run_training(
    train, valid, column_features, column_targets, fold, seed, verbose=False, **kwargs
):

    seed_everything(seed)
    writer = SummaryWriter(f"try_0_42param_schnet_Dguo_mixmape_{exp_name}_{fold}")

    # print(train['normalized_stopping_power'].head())

    train_dataset = SolidSchnet(train, schnet_feat, column_features)
    valid_dataset = SolidSchnet(valid, schnet_feat, column_features)

    # print(train_dataset[100])

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=_collate_aseatoms,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    validloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=_collate_aseatoms,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    model_sarasa = Model(num_features=39, num_targets=1)

    model = spk.atomistic.Atomwise(
        n_in=None, aggregation_mode="avg", outnet=model_sarasa
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.3, div_factor=1e3,
    #    max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)

    loss_fn = CompoundMAPELoss(ratio=10000).to(DEVICE)
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    # todo el guardado de los resultados se puede mover a kfold que si tiene info de los indices
    # oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf

    for epoch in range(EPOCHS):

        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)

        valid_loss, valid_preds, val_mape = valid_fn(
            model, scheduler, loss_fn, validloader, DEVICE
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", valid_loss, epoch)
        writer.add_scalar("Mape/", val_mape, epoch)

        if verbose:
            print(
                f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}, val_loss:{valid_loss}, val_mape:{val_mape}"
            )

        if valid_loss < best_loss:

            best_loss = valid_loss
            oof = valid_preds

            torch.save(model.state_dict(), f"../results/FOLD{fold}_{exp_name}.pth")

        elif EARLY_STOP == True:

            early_step += 1
            if early_step >= early_stopping_steps:
                break

    # --------------------- PREDICTION---------------------

    ##  testdataset = TestDataset(X_test)
    # testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    #     model = Model(
    #          num_features= X_train.shape[1] ,
    #         num_targets=  y_train.shape[1],
    #         hidden_size=hidden_size,**kwargs
    #     )

    #     model.load_state_dict(torch.load(f"../results/FOLD{fold}_{exp_name}.pth"))
    # model.to(DEVICE)

    # predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    # predictions = inference_fn(model, testloader, DEVICE)
    writer.close()
    return oof  # ,  predictions
