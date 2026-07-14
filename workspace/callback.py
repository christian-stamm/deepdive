from .state import State


class Callback:

    def on_epoch_begin(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        pass

    def on_train_begin(self, state: State):
        pass

    def on_train_end(self, state: State):
        pass

    def on_val_begin(self, state: State):
        pass

    def on_val_end(self, state: State):
        pass
