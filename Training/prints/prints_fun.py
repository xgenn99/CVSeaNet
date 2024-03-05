def prints(real_conv, resolution, in_channels, batch_size, epochs, lr, gamma, alpha, w1, w2, score_threshold):
    print(f"The backbone is complex valued: {not real_conv}")
    print(f"in_channels: {in_channels} | resolution: {resolution} | batch_size: {batch_size} | epochs: {epochs} | lr: {lr} ")
    print(f"loss gamma: {gamma}| loss alpha: {alpha}| weight preds: {w1} | weight off: {w2}")
    print(f"score threshold for acc: {score_threshold}")
