
###### tags: `Pytorch`


# Save and load

### 兩種保存方法

```
torch.save(net1, 'net.pkl')
torch.save(net1.state_dict(), 'net_params.pkl')
```

1.是保存整個模型
2.保存Node 參數(較快 但提取需與 原本的架構相同)


### 提取方法

```
net2 = torch.load('net.pkl')
prediction = net2(x)
```
對應第一種保存方法，就是直接拿

```
net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    
net3.load_state_dict(torch.load('net_params.pkl'))
prediction = net3(x)
```
對應第二種方法比較麻煩但比較快一點，需要再複寫一次架構


驗證:
![](https://i.imgur.com/nc25rYr.png)


# 批次訓練
```
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)
```

