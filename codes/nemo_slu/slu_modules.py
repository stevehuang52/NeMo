from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.nlp.modules.common.transformer import TransformerDecoder
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins
from nemo.core.classes.module import NeuralModule


class SLUTransformerDecoder(NeuralModule, Exportable, adapter_mixins.AdapterModuleMixin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.net = TransformerDecoder(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)
