import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# Logging parameters
RUN_NAME = "GPT_XTTS_LJSpeech_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = r"C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\main\output"

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 12  # set here the batch size
GRAD_ACUMM_STEPS = 21  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path=r"C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder",
    meta_file_train=r"C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\output.csv",
    language="en",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Define the path where XTTS v1.1.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v1.1_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, DVAE_CHECKPOINT_LINK.split("/")[-1])
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, MEL_NORM_LINK.split("/")[-1])

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


# Download XTTS v1.1 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, TOKENIZER_FILE_LINK.split("/")[-1])  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, XTTS_CHECKPOINT_LINK.split("/")[-1])  # model.pth file

# download XTTS v1.1 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v1.1 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )


# Training sentences generations
SPEAKER_REFERENCE = [
    r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\1.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\2.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\3.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\4.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\5.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\6.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\7.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\8.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\9.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\10.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\11.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\12.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\13.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\14.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\15.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\16.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\17.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\18.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\19.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\20.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\21.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\22.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\23.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\24.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\25.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\26.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\27.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\28.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\29.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\30.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\31.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\32.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\33.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\34.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\35.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\36.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\37.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\38.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\39.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\40.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\41.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\42.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\43.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\44.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\45.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\46.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\47.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\48.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\49.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\50.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\51.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\52.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\53.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\54.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\55.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\56.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\57.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\58.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\59.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\60.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\61.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\62.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\63.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\64.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\65.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\66.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\67.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\68.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\69.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\70.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\71.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\72.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\73.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\74.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\75.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\76.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\77.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\78.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\79.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\80.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\81.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\82.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\83.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\84.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\85.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\86.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\87.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\88.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\89.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\90.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\91.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\92.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\93.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\94.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\95.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\96.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\97.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\98.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\99.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\100.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\101.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\102.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\103.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\104.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\105.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\106.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\107.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\108.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\109.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\110.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\111.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\112.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\113.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\114.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\115.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\116.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\117.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\118.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\119.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\120.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\121.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\122.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\123.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\124.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\125.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\126.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\127.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\128.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\129.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\130.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\131.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\132.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\133.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\134.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\135.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\136.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\137.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\138.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\139.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\140.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\141.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\142.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\143.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\144.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\145.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\146.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\147.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\148.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\149.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\150.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\151.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\152.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\153.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\154.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\155.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\156.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\157.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\158.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\159.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\160.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\161.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\162.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\163.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\164.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\165.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\166.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\167.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\168.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\169.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\170.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\171.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\172.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\173.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\174.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\175.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\176.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\177.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\178.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\179.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\180.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\181.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\182.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\183.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\184.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\185.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\186.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\187.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\188.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\189.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\190.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\191.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\192.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\193.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\194.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\195.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\196.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\197.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\198.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\199.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\200.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\201.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\202.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\203.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\204.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\205.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\206.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\207.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\208.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\209.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\210.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\211.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\212.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\213.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\214.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\215.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\216.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\217.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\218.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\219.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\220.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\221.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\222.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\223.wav', r'C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs\224.wav'# speaker reference to be used in training test sentences
]
LANGUAGE = config_dataset.language


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        # tokenizer_file="/raid/datasets/xtts_models/vocab.json", # vocab path of the model that you want to fine-tune
        # xtts_checkpoint="https://huggingface.co/coqui/XTTS-v1/resolve/hifigan/model.pth",
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=8194,
        gpt_start_audio_token=8192,
        gpt_stop_audio_token=8193,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "I don't want to run away from this. I ran from my mom's death for so long... I misunderstood it... Everything was just too painful for me to handle. But... I never want to be left in the dark again! ...That's probably why my Persona is a little bit special. Because I want to learn the truth. That's how I really feel! Well, I've come this far... I'll follow you wherever you go, on my own two feet!",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()