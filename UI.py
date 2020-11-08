import pygame
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
import button
import os
from PIL import Image
import numpy as np
from skimage import transform

pygame.init()

item_list = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

model = keras.models.load_model('model.h5')
font_30 = pygame.font.SysFont("arial", 30)

def load(filename):
  np_image = Image.open(filename)
  np_image = np.array(np_image).astype('float32') / 255
  np_image = transform.resize(np_image, (512, 384, 3))
  np_image = np.expand_dims(np_image, axis=0)
  return np_image

def process_result(li):
  temp = list()
  for item in li:
    for num in item:
      temp.append(num)
  li = temp.copy()
  highest_val = max(li)
  index = li.index(highest_val)
  return item_list[index]

screen_x = 520
screen_y = 720
window = pygame.display.set_mode((screen_x, screen_y))
pygame.display.set_caption("Trash Sorting")
icon = pygame.image.load('trash_icon.png')
pygame.display.set_icon(icon)
trash_image = pygame.image.load('trash.png')

select_btn = button.button(screen_x/2 - 150/2 + 100, screen_y/2+100, 150, 64, (255, 255, 0), 'Select Image')
again_btn = button.button(screen_x/2 - 150/2 + 100, screen_y/2+100, 150, 64, (255, 255, 0), 'Again')
quit_btn = button.button(screen_x/2 - 150/2 - 100, screen_y/2+100, 150, 64, (255, 255, 0), 'QUIT')

execute = True
classify = False

temp_img = None
img_num = None

while execute:
    window.fill((128, 128, 128))
    window.blit(trash_image, (screen_x/2 - 256/2, 100))
    select_btn.draw(window, True)
    quit_btn.draw(window, True)
    pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            execute = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if select_btn.is_over(pos):
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename()
                if file_path != "":
                    img_num = load(file_path)
                    temp_img = pygame.image.load(file_path)
                    print(file_path)
                    classify = True
            if quit_btn.is_over(pos):
                execute = False
    while classify:
        # Resolution: 512 * 384
        temp_img = pygame.transform.scale(temp_img, (256, 192))
        window.fill((128, 128, 128))
        again_btn.draw(window, True)
        quit_btn.draw(window, True)
        window.blit(temp_img, (screen_x / 2 - 256 / 2, 100))
        result = process_result(model.predict(img_num))
        text_result = font_30.render(result, True, (255, 255, 255))
        window.blit(text_result, (screen_x/2 - text_result.get_width() / 2, 300))
        pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                execute = False
                classify = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if again_btn.is_over(pos):
                    classify = False
                if quit_btn.is_over(pos):
                    classify = False
                    execute = False
        pygame.display.update()

    pygame.display.update()
