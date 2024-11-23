# Key Changes Made

- **Separated Paths**:
  - Separated `pretrained_path` (for fine-tuning) from `checkpoint_path` (for resuming training).

- **Learning Rate Adjustment**:
  - Added learning rate reduction for fine-tuning.

- **Error Handling**:
  - Added proper error handling for missing model files.

- **Checkpoint Management**:
  - Created new checkpoint paths for each run instead of overwriting.

- **Logging**:
  - Added clear logging of what type of training is being performed.

- **File Existence Verification**:
  - Added verification of file existence before starting training.

## Training Scenarios

This setup allows for three distinct scenarios:

1. **Fresh Training**:
   - Call: `train(cfg)`

2. **Fine-Tuning**:
   - Call: `train(cfg, pretrained_path=path)`

3. **Resume Training**:
   - Call: `train(cfg, checkpoint_path=path)`

## Additional Fine-Tuning Features (Optional)

Would you like me to add any of the following fine-tuning features?

- Freezing certain layers
- Gradual unfreezing
- Different learning rates for different parts of the network
- Custom fine-tuning schedules