pipelines = []
augmentations = [1, 2, 3, 4, 5]
augmentations_start = ["start"]
augmentations_end = ["end"]
final_aug = augmentations_start + augmentations + augmentations_end
pipelines.append(final_aug)
for k in range(4):
    augmentations = [augmentations[-1]] + augmentations[0:-1]
    final_augs = augmentations_start + augmentations + augmentations_end
    pipelines.append(final_augs)

print(pipelines)