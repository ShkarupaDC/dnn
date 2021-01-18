class Callback():

    def on_train_start(self, model, **kwargs):
        return 

    def on_train_end(self, model, **kwargs):
        return 
    
    def on_epoch_start(self, model, **kwargs):
        return 

    def on_epoch_end(self, model, **kwargs):
        return 

    def on_train_epoch_start(self, model, **kwargs):
        return 

    def on_train_epoch_end(self, model, **kwargs):
        return 

    def on_valid_epoch_start(self, model, **kwargs):
        return 

    def on_valid_epoch_end(self, model, **kwargs):
        return 

    def on_train_step_start(self, model, **kwargs):
        return 

    def on_train_step_end(self, model, **kwargs):
        return 
    
    def on_valid_step_start(self, model, **kwargs):
        return 
    
    def on_valid_step_end(self, model, **kwargs):
        return 


class CallbackRunner():

    def __init__(self, model, callbacks):
        
        self.model = model
        self.callbacks = callbacks

    def __call__(self, state, **kwargs):

        for callback in self.callbacks:
            _ = getattr(callback, 'on_' + state)(self.model, **kwargs)