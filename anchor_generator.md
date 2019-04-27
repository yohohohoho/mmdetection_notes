## 总体结构
anchor的初始化主要依赖于 base_size, scales, ratios。
计算代码如下：
```
def gen_base_anchors(self):
    w = self.base_size
    h = self.base_size
    if self.ctr is None:
        x_ctr = 0.5 * (w - 1)
        y_ctr = 0.5 * (h - 1)
    else:
        x_ctr, y_ctr = self.ctr

    h_ratios = torch.sqrt(self.ratios)
    w_ratios = 1 / h_ratios
    if self.scale_major:
        ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
        hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
    else:
        ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
        hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

    base_anchors = torch.stack(
        [
            x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
        ],
        dim=-1).round()

    return base_anchors
```
w,h: base_size的大小
ctr: 中心坐标
h_ratios，w_ratios： ratios的平方根
ws, hs: 对应原图长宽
base_anchors: 相对于中心的偏移

以如下参数测试：
```
anchor_base = 4
anchor_scales = [8, 16, 32]
anchor_ratios = [0.5, 1.0, 2.0]
```
结果为：
```
w:4
h:4
x_ctr:1.5
y_ctr:1.5
h_ratios:tensor([0.7071, 1.0000, 1.4142])
w_ratios:tensor([1.4142, 1.0000, 0.7071])
ws:tensor([ 45.2548,  90.5097, 181.0193,  32.0000,  64.0000, 128.0000,  22.6274,
         45.2548,  90.5097])
hs:tensor([ 22.6274,  45.2548,  90.5097,  32.0000,  64.0000, 128.0000,  45.2548,
         90.5097, 181.0193])
-------------------------------------------
tensor([[-21.,  -9.,  24.,  12.],
        [-43., -21.,  46.,  24.],
        [-89., -43.,  92.,  46.],
        [-14., -14.,  17.,  17.],
        [-30., -30.,  33.,  33.],
        [-62., -62.,  65.,  65.],
        [ -9., -21.,  12.,  24.],
        [-21., -43.,  24.,  46.],
        [-43., -89.,  46.,  92.]])
```
