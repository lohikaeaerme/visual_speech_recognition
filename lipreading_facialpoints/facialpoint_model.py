from torch import nn


class facialpoint_model(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(facialpoint_model,self).__init__()
        hidden_dim_1 = int(input_dim / 20)
        hidden_dim_2 = int(input_dim / 200)
        #hidden_dim_3 = int(input_dim / 20)

        layer_1 = nn.Linear(input_dim, hidden_dim_1)
        layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        #layer_3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        layer_4 = nn.Linear(hidden_dim_2, output_dim)

        self.model = nn.Sequential(
            layer_1,
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            layer_2,
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            #layer_3,
            #nn.ReLU(),
            layer_4
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
