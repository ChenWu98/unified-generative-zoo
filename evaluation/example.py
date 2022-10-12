# Created by Chen Henry Wu


class Evaluator(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

    def evaluate(self, images, model, weighted_loss, losses, data, split):
        """

        Args:
            images: list of images (or None), or list of tuples of images (or tuples of None)
            model: model to evaluate
            weighted_loss: list of scalar tensors
            losses: dictionary of lists of scalar tensors
            data: list of dictionary
            split: str

        Returns:

        """
        assert split in ['eval', 'test']

        summary = {}
        # Add metrics here.

        return summary

