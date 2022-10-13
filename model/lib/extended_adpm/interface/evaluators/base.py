import logging


class Evaluator(object):
    def __init__(self, options: dict):
        r""" Evaluate models
        """
        self.options = options

    def evaluate_train(self, it):
        r""" Evaluate during training
        Args:
            it: the iteration of training
        """
        for key, option in self.options.items():
            period = option.period
            kwargs = option.get('kwargs', {})
            if it % period == 0:
                logging.info(f'evaluate {key}')
                eval(f'self.{option.fn}')(it=it, **kwargs)

    def evaluate(self, it=None):
        r"""
        Args:
            it: the iteration when the evaluated models is saved
        """
        if it is None:
            it = 0
        for key, option in self.options.items():
            logging.info(f'evaluate {key}')
            kwargs = option.get('kwargs', {})
            eval(f'self.{option.fn}')(it=it, **kwargs)
