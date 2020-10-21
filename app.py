from operator import ge
from re import sub
import streamlit as st
from torch import mode
from preprocess import check_path, get_image_dataset, get_dataloader, get_model

st.title('Easy AI  🚀🚀🚀')
st.header('Streamlit 🔥 + PyTorch ❤️')
st.text('Train Deep Learning Image classifcation Models without any code ')

image_data_path = st.text_input('Enter image data path',
                                value='Example: /Users/adam/data/cats_vs_dogs')

n_epochs = st.number_input('Number of epochs', min_value=1, value=10)

submit = st.button('Submit')


def train_loop(model, dataloader, n_epochs, optimizer, criterion, data_size):
    import time
    pbar = st.progress(n_epochs)
    print(n_epochs)

    for epoch in range(n_epochs):
        running_loss = 0.0
        for data in dataloader:
            images, labels = data

            optimizer.zero_grad()
            outputs = model(images)
            # print('labels.size, outputs.size', labels.size(), outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / data_size
        print(f'{epoch} Loss: {epoch_loss:.4f}')
        pbar.progress(epoch + 1)


if submit:
    assert isinstance(n_epochs, int)
    if not check_path(image_data_path):
        st.error(f'{image_data_path} doesn\'t exist!')

    image_ds = get_image_dataset(image_data_path)
    dataloader = get_dataloader(image_ds, batch_size=8)
    data_size = len(image_ds)
    model, criterion, optimizer = get_model(image_ds)

    train_loop(model, dataloader, n_epochs, optimizer, criterion, data_size)
