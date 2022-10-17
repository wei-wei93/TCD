from tkinter import *
import tkinter.ttk as ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog as fd
from two_loops_for_GUI import bayesianMattingTwoLoops


# Graphic User Interface
def on_click(input_img, trimap, gt, sigma_for_gsn, min_fg_pixels,
             min_bg_pixels, output_alpha_matte,
             output_foreground, output_background):
  if input_img.get() == '' or trimap.get() == '' or sigma_for_gsn.get() == '' \
                        or min_fg_pixels.get() == '' \
                        or min_bg_pixels.get() == '':
    tip['fg'] = 'red'
    tip['text'] = f'{"Please fill in all the * blanks!"}'
  elif output_alpha_matte.get() == '' and output_foreground.get() == '' \
                                   and output_background.get() == '':
    tip['fg'] = 'red'
    tip['text'] = f"At least one output file path\n must be filled in!"
  else:
    tip['fg'] = 'green'
    tip['text'] = ""
    main_window.update()
    bayesianMattingTwoLoops(input_img.get(), trimap.get(), gt.get(),
                            float(sigma_for_gsn.get()),
                            int(min_fg_pixels.get()),
                            int(min_bg_pixels.get()), output_alpha_matte.get(),
                            output_foreground.get(), output_background.get(),
                            main_window, bar, progress_label, running_time,
                            psnr_label, ssim_label)
    tip['text'] = f'{"Done!"}'
    showinfo(message='The progress completed!')


def open_file(entry_variable):
  filetypes = (
        ('png files', '*.png'),
        ('All files', '*.*')
  )
  filename = fd.askopenfilename(
    title='Open',
    initialdir='./',
    filetypes=filetypes
  )
  entry_variable.set(filename)


def write_file(entry_variable):
  filetypes = (('png files', '*.png'), ('All files', '*.*'))
  filename = fd.asksaveasfilename(title='Save as', initialdir='./',
                                  defaultextension='.png', filetypes=filetypes)
  entry_variable.set(filename)

main_window = Tk()
# main_window.title('Our_first_company, Yeah!')
# set StringVar placeholders
input_img = StringVar()
trimap = StringVar()
gt = StringVar()
sigma_for_gsn = StringVar()
min_fg_pixels = StringVar()
min_bg_pixels = StringVar()
output_alpha_matte = StringVar()
output_foreground = StringVar()
output_background = StringVar()

# Labels
# title
title = Label(main_window, text='Bayesian Matting Program',
              font=("Arial", 18, "bold"))
title.grid(row=0, column=0, columnspan=3, pady=20, sticky='ew')

# input label
Label(main_window, text="Inputs").grid(row=1, column=1)
Label(main_window, text="Input image file path*").grid(row=2, column=0)
Label(main_window, text="Trimap file path*").grid(row=3, column=0)
Label(main_window, text="Ground Truth file path (optional)").grid(row=4,
                                                                  column=0)
Label(main_window, text="Sigma for Gaussian* (float>0)").grid(row=5, column=0)
Label(main_window, text="Minimum foreground pixels* (int>0)").grid(row=6,
                                                                   column=0)
Label(main_window, text="Minimum background pixels* (int>0)").grid(row=7,
                                                                   column=0)

# output label
Label(main_window, text="Outputs").grid(row=9, column=1)
Label(main_window, text="Output alpha matte file path").grid(row=10, column=0)
Label(main_window, text="Output foreground file path").grid(row=11, column=0)
Label(main_window, text="Output background file path").grid(row=12, column=0)

# Text Entrys
# input entrys
input_img_entry = Entry(main_window, textvariable=input_img, width=50,
                        borderwidth=5).grid(row=2, column=1)
trimap_entry = Entry(main_window, textvariable=trimap, width=50,
                     borderwidth=5).grid(row=3, column=1)
gt_entry = Entry(main_window, textvariable=gt, width=50,
                 borderwidth=5).grid(row=4, column=1)
sigma_for_gsn_entry = Entry(main_window, textvariable=sigma_for_gsn, width=50,
                            borderwidth=5).grid(row=5, column=1)
min_fg_pixels_entry = Entry(main_window, textvariable=min_fg_pixels, width=50,
                            borderwidth=5).grid(row=6, column=1)
min_bg_pixels_entry = Entry(main_window, textvariable=min_bg_pixels, width=50,
                            borderwidth=5).grid(row=7, column=1)

# output entries
output_alpha_matte_entry = Entry(main_window, textvariable=output_alpha_matte,
                                 width=50, borderwidth=5).grid(row=10,
                                                               column=1)
output_foreground_entry = Entry(main_window, textvariable=output_foreground,
                                width=50, borderwidth=5).grid(row=11, column=1)
output_background_entry = Entry(main_window, textvariable=output_background,
                                width=50, borderwidth=5).grid(row=12, column=1)

# open buttons
photo = PhotoImage(file=r".\openfile_icon.png")

# input openfile
Button(main_window, text='Clike me', image=photo, command=lambda
       arg1=input_img: open_file(arg1)).grid(row=2, column=2, padx=8)
Button(main_window, text='Clike me', image=photo,
       command=lambda arg1=trimap: open_file(arg1)).grid(row=3, column=2)
Button(main_window, text='Clike me', image=photo,
       command=lambda arg1=gt: open_file(arg1)).grid(row=4, column=2)

# output openfile
Button(main_window, text='Clike me', image=photo, command=lambda
       arg1=output_alpha_matte: write_file(arg1)).grid(row=10, column=2)
Button(main_window, text='Clike me', image=photo, command=lambda
       arg1=output_foreground: write_file(arg1)).grid(row=11, column=2)
Button(main_window, text='Clike me', image=photo, command=lambda
       arg1=output_background: write_file(arg1)).grid(row=12, column=2)

# generate button
Button(main_window, text="Generate", fg='green', command=lambda
       arg1=input_img, arg2=trimap, arg3=gt, arg4=sigma_for_gsn,
       arg5=min_fg_pixels, arg6=min_bg_pixels, arg7=output_alpha_matte,
       arg8=output_foreground,
       arg9=output_background: on_click(arg1, arg2, arg3, arg4, arg5, arg6,
                                        arg7, arg8, arg9)).grid(row=8,
                                                                column=1,
                                                                pady=10)

# show warnings
tip = Label(main_window, text='')
tip.grid(row=8, column=0)

# show progress bar
bar = ttk.Progressbar(main_window, orient=HORIZONTAL, length=360,
                      mode='determinate')
bar.grid(row=13, column=1, pady=20)
progress_label = Label(main_window,
                       text=f"Current Progress: {round(bar['value'], 1)}%")
progress_label.grid(row=13, column=0)

# show running time
Label(main_window, text="Running Time: ").grid(row=14, column=0)
running_time = Label(main_window, text=f"0.0sec")
running_time.grid(row=14, column=1)

# show performance
Label(main_window, text="Performance").grid(row=15, column=0)

Label(main_window, text="PSNR: ").grid(row=16, column=0)
psnr_label = Label(main_window, text=f"0.0dB")
psnr_label.grid(row=16, column=1)

Label(main_window, text="SSIM: ").grid(row=17, column=0)
ssim_label = Label(main_window, text=f"0.0")
ssim_label.grid(row=17, column=1)

main_window.mainloop()
