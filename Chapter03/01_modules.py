import torch
import torch.nn as nn

class OurModule(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_classes: int,
                 dropout_prob: float=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
                                nn.Linear(in_features=num_inputs,
                                        out_features=5),
                                nn.ReLU(),
                                nn.Linear(in_features=5,
                                        out_features=20),
                                nn.ReLU(),
                                nn.Linear(in_features=20,
                                        out_features=num_classes),
                                nn.Dropout(p=dropout_prob),
                                nn.Softmax(dim=1)
                                 )

    def forward(self, x):
        return self.pipe(x)

if __name__ == "__main__":
    
    # Instantiate the net
    net = OurModule(num_inputs=2, num_classes=3)
    
    # Print the architecture
    print(net)
    
    # Create input of dim (1,2), feed it to the network and print the output
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(out)
    print("Cuda's availability is %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Data from cuda: %s" % out.to('cuda'))
