SAMPLING_RATIO=0.4

EPOCHS=4
BATCH_SIZE=20

IMAGE_SIZE=(224,244)

IMG_TRAIN_PATH='DogFaceNet/after_4_bis/*/*.jpg'
EMBED_IMG_PATH='reference_images/*.jpg'
LOG_DIR='logdir_train'

DEVICE="cpu"
MODEL_SAVEPATH="modelsave"