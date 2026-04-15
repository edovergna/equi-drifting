import pytorch_lightning as pl

from model.lit_modules import QM9DataModule, LitFlowMatching

def main():
    pl.seed_everything(42)

    datamodule = QM9DataModule()
    model = LitFlowMatching()
    callbacks = []

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
