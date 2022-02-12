import cv2
import os
import shutil

from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from litmodels import LitModel
from data import PatchDataModule, prepare_data

from data import read_image_tensor, write_image_tensor, ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from PIL import Image

from video2frames import FrameCapture
from frames2video import generate_video

data_path = Path('data/current')
# data is excpected to be in folders:
# data_path /
#           input
#           target
#           mask (optional)

model_save_path = data_path / 'models'
inputpath = Path('inputfiles')
#outputpath = Path('outputfiles')

# erledigt nacheinander das zerlegen in Einzelbilder, Training, Style transfer und wieder zusammensetzen
seq_length = 40                                         # Anzahl an Trainingsepochen pro Satz, nach jedem Satz wird ein Video generiert und gefragt, ob das Ergebnis zufriedenstellt
num_sequences = 5                                       # maximale Anzahl an Sätzen

def doInputStuff():
    cwd = os.getcwd()
    txt = ' '
    while txt ==' ':
        print('liegt der input als video(v) oder in einzelbildern (f) vor?')
        txt = input()
        if txt == 'v':
            print('Name der videodatei')
            videopath = os.path.join(inputpath,input())  #expects video in current input-directory
            imagepath = data_path
            fps = FrameCapture(videopath,imagepath/'input')
        elif txt =='f':
            print('gewünschte framerate')
            fps = int(input())
        else:
            txt = ' '
    os.chdir(cwd)

    return fps

if __name__ == "__main__":
    framerate = doInputStuff()
    print('name des Zielvideos (ohne Dateiendung)')
    videoname = input()
    print('warten auf keyframes')
    input()
    #########################################training#######################################################################################################train
    print('train - anfang')
#    framerate = 30
# das Training geschieht jetzt in mehreren Schritten
    for i in range(num_sequences):
        logger = TensorBoardLogger(Path(), 'lightning_logs')

        profiler = pl.profiler.SimpleProfiler()

        callbacks = []

        train_image_dd = prepare_data(data_path)

        dm = PatchDataModule(train_image_dd,
                            patch_size=2**6,
                            batch_size=2**3,
                            patch_num=2**6)

        model = LitModel( use_adversarial=True)

        # uncomment next line to start from latest checkpoint
        if i>0:
            model = LitModel.load_from_checkpoint(model_save_path/"latest.ckpt")
        
        trainer = pl.Trainer(
            gpus=-1, 
            precision=16,
            max_epochs=seq_length,
            log_every_n_steps=8,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            check_val_every_n_epoch=seq_length/2,
            reload_dataloaders_every_n_epochs=1,
            profiler=profiler,
            logger=logger,
            callbacks=callbacks,
            # fast_dev_run=True,
        )

        trainer.fit(model, dm)

        trainer.save_checkpoint(model_save_path/"latest.ckpt")
        torch.save(model.generator, model_save_path/"generator.pt")
        torch.save(model.discriminator, model_save_path/"discriminator.pt")
        print('#9')
        #################################styletransfer########################################################################################################################style transfer
        output_dir = data_path/'output'
        input_dir = data_path/'input'

        # Change these depending on your hardware, has to match training settings
        device = 'cuda' 
        dtype = torch.float16 

        generator = torch.load(model_save_path/"generator.pt")
        generator.eval()
        generator.to(device, dtype)


        # TODO batch size, async dataloader
        file_paths = [file for file in input_dir.iterdir()]

    #eine Erhöhung der batch-size beschleunigt den Prozess massiv, ABER: zu große batches resutlieren in runtime error: CUDA out of memory, daher
    #muss man hier in Abhängigkeit von der AUflösung der Bilder die batchsize individuell wählen
    #im Zweifel erstmal 'batch_size': 1 und dann schrittweise erhöhen
        params = {'batch_size': 8,
                'num_workers': 8,
                'pin_memory': True}

        dataset = ImageDataset(file_paths,)
        loader = DataLoader(dataset, **params)

        # TODO multiprocess and asynchronous writing of files

        with torch.no_grad():
            for inputs, names in tqdm(loader):
                inputs = inputs.to(device, dtype)
                outputs = generator(inputs)
                del inputs
                for j in range(len(outputs)):
                    write_image_tensor(outputs[j], output_dir/names[j])
                del outputs

        ######################################################frames2video#############################################################################################

        vname = videoname+str(seq_length*(i+1))+'eps'
        print(vname)
    #    print(data_path / 'output',videoname,framerate)
        generate_video(data_path / 'output',vname+'.mp4',framerate)
        print(seq_length*(i+1),' Epochen trainiert, zufrieden?')
        input()
        #TODO: Funktionalität hinzufügen 

