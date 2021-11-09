#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


##### Changed
class LR(nn.Module):
    def __init__(self, num_classes, feature_dims):
        super(LR, self).__init__()
        self.linear = nn.Linear(feature_dims, num_classes)

    def forward(self, x):
        # Assumes x of shape (N, x_dim)
        # return F.log_softmax(self.linear(x), dim=1)
        return self.linear(x)   # returns logits


# class LR(nn.Module):
#     def __init__(self, num_classes, feature_dims):
#         super(LR, self).__init__()
#         self.linear1 = nn.Linear(feature_dims, layerdims[0])
#         self.linear2 = nn.Linear(layerdims[0], layerdims[1])
#         self.linear3 = nn.Linear(layerdims[1], num_classes)

#     def forward(self, x):
#         # Assumes x of shape (N, x_dim)
#         # return F.log_softmax(self.linear(x), dim=1)
#         return self.linear3(self.linear2(self.linear1(x)))   # returns logits



def convert_to_batches(X, Y, batch_size):
    # reshape X from N,dim to B, N/batch_size, dim
    random_indices = torch.randperm(X.shape[0])
    X = X[random_indices, :]
    Y = Y[random_indices]
    batches = []
    whole_batches = X.shape[0] // batch_size
    temp_X = X[:whole_batches*batch_size, :].reshape(whole_batches, batch_size, X.shape[-1])
    temp_Y = Y[:whole_batches*batch_size].reshape(whole_batches, batch_size)
    for el1, el2 in zip(temp_X, temp_Y):
        batches.append((el1, el2))
    batches.append((X[whole_batches*batch_size:, :], Y[whole_batches*batch_size:]))
    return batches


def calc_accuracy(pred, target):
    # assumes pred of shape NxC with a log probability for every class C, and target of shape N
    max_vals, max_indices = torch.max(pred, 1)
    n = max_indices.size(0)
    train_acc = (max_indices == target).sum(dtype=torch.float32) / n
    return train_acc


def plot_results(train_loss, val_loss):
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.show()

#%%
##### Changed
def train(X_train, Y_train, query_data, query_label, batch_size=1024, epochs=200, learning_rate=0.08, return_model=False, layerdims=[]):

    #%%
    # learning_rate = 0.08
    # epochs = 500

    X_test = query_data
    Y_test = query_label

    torch.manual_seed(0)

    batches = convert_to_batches(X_train, Y_train, batch_size)

    num_classes = Y_train.unique().size(0)

    ##### Changed
    # model = LR(num_classes, X_train.shape[-1], layerdims).to(X_train.device)
    model = LR(num_classes, X_train.shape[-1]).to(X_train.device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # train_loss = []
    # val_loss = []
    val_acc = []

    for epoch in range(epochs):
        # mb_loss = 0

        for mini_batch in batches:
            X = mini_batch[0]
            Y = mini_batch[1]
            
            logits = model(X)
            loss = loss_function(logits, Y)

            # mb_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # train_loss.append(mb_loss/len(batches))

        # validation monitoring
        with torch.no_grad():
            logits = model(X_test)
            # loss1 = loss_function(logits, Y_test)
            # val_loss.append(loss1.item())
            acc1 = calc_accuracy(logits.softmax(dim=1), Y_test)
            val_acc.append(acc1)

    # plt.plot(train_loss, label='train')
    # plt.plot(val_loss, label='val')
    # plt.legend()
    # plt.show()
    # plt.plot(val_acc, label='acc')
    # plt.legend()
    # plt.show()
    # print(max(val_acc))
    # print(acc1.item())

    if return_model:
        return torch.stack(val_acc).max(), model
    else:
        return torch.stack(val_acc).max()
