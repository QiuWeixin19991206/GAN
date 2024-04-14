import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from scipy.misc import toimage
from PIL import Image#Image替代toimage
import glob
from GAN import Generator, Discriminator
from dataset import make_anime_dataset

assert tf.__version__.startswith('2.')#用于检查当前使用的TensorFlow版本是否以’2.'开头。如果不是以’2.'开头，则会引发一个AssertionError异常
tf.random.set_seed(1234)
np.random.seed(1234)

# 把多张结果图拼接为一张并保存
def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    # toimage(final_image).save(image_path)

    image_pil = Image.fromarray(np.uint8(final_image))# 将numpy数组转换为PIL图像对象
    image_pil.save(image_path)# 保存图像到文件

def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)

def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 1.treat real image as real
    # 2.treat generated image as fake
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    loss = d_loss_fake + d_loss_real
    return loss

def g_loss_fc(generator, discriminator, batch_z, batch_x, is_training):

    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)#希望生成器接近真实 所以为1

    return loss

def main():

    # hyper parameters
    z_dim = 100
    epochs = 300000
    batch_size = 128
    learning_rate = 1e-3
    is_training = True

    img_path = glob.glob(r'F:\qwx\学习计算机视觉\机器学习\tensorflow2\整理\测试实战\GAN\data\anime_face\*.jpg')#读取图片路径 *.jpg读取jpg文件
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample).numpy())

    dataset = dataset.repeat()#对数据集进行重复操作，以便在训练过程中能够多次使用数据
    db_iter = iter(dataset)#创建了一个迭代器 db_iter，这样就可以逐个元素地遍历数据集

    generator = Generator()
    generator.build(input_shape= (None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape= (None, 64, 64, 3))

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        #虚假图片
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)#像素是-1~1
        # 真实图片
        batch_x = next(db_iter)

        #train D
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fc(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, ' d_loss: ', float(d_loss), 'g_loss:', float(g_loss))

            #保存图片 用于检测
            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('photo', 'face_%d.png'%epoch)
            save_result(fake_image.numpy(), 10, image_path=img_path, color_mode='p')#color_mode='p'彩色图片



if __name__ =='__main__':
    tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)
    print("is_gpu: ", tf.test.is_gpu_available())
    main()









