import sys, os, time
from data.util import load
from metrics.similarity import mmd
from metrics.conditional import mrr

PATH = os.path.dirname(os.path.realpath(__file__))

model = sys.argv[1]
split = int(sys.argv[2])

assert model in ['PGAN']
assert split in [1,2,3,4,5]

train = load('train',split)
test = load('test', split)
val = load('val', split)

if model == 'PGAN':
    from models.PGAN import Trainer
    from models.PGAN import Trainer
    from models.config import PGAN_config as model_config

trainer = Trainer(
    batch_size = model_config['batch_size'],
    plot_every = 10,
    checkpoint_every = 10,
    train_data = train,
    test_data = test,
    val_data = val,
    path = '{}/split_{}/{}'.format(PATH,split,model),
    config = model_config,
    pad_sequence = model_config['pad_sequence'],
    pad_labels = model_config['pad_labels'],
    short_val = model == 'transformer',
)

trainer.train(epochs=500, progress=True, restore=False)
