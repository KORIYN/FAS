This is the official implementation based on pytorch of the FAS method.


## Usage
We provide an example demonstrating how FSTA is integrated into the model to operate on a shallow layer of intermediate features during source-domain training. The complete implementation will be publicly released in the near future.

```python
import torch
from .frequency_topk import Frequency_TopK

bsz, c, h, w = 128, 384, 14, 14
immediate_feature = torch.rand(bsz, c, h, w)

output_feature = Frequency_TopK(immediate_feature)


