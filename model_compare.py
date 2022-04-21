import torch

def test():
    lmd = torch.load('ltest.pt')
    md = torch.load('test.pt')
    mk = list(md.keys())
    mv = list(md.values())
    lmv = list(lmd.values())
    lmk = list(lmd.keys())
    print(lmk)
    print(mk)
    print(lmk)
    tensor1 = lmv[11].cpu().float()
    tensor2 = mv[7]
    print(tensor1)
    print(tensor2)
    print(torch.equal(tensor1, tensor2))


if __name__ == '__main__':
    test()
