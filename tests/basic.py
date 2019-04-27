import torch


class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()

    def forward(self, input1, input2):
        return input1 + input2


test_module = TestModule()

example = (torch.rand(1, 4), torch.rand(1, 4))
traced_script_module = torch.jit.trace(test_module, example)
