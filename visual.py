import matplotlib.pyplot as plt
import PySimpleGUI as sg
import os
from PIL import Image, ImageTk
import io
from result_dir import make_dir_if_not_exists
from settings import DIR_RESULT, DIR_PLOT


def show_results(y_test, y_pred, estimator):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(range(len(y_test)), y_test, c="red", label='Actual')
    ax.plot(range(len(y_test)), y_pred, c="green", label='Prediction')
    estm_name = str(estimator)
    plt.title(estm_name)
    plot_path = f'{DIR_RESULT}/{DIR_PLOT}/'
    make_dir_if_not_exists(plot_path)
    re_ch = ['"',"'",'\n','[',']','(',')',' ', ',', ':', '-', '.']
    for ch in re_ch:
        estm_name = estm_name.replace(ch, '_')
    plt.legend()
    fig.savefig(f'{plot_path}{estm_name}.png')
    plt.close(fig)
    # plt.show()

def get_img_data(f, maxsize=(1200, 850), first=False):
    """Generate image data using PIL
    """
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:  # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def show_image(path):
    path = f'{path}/{DIR_PLOT}'
    # Get the folder containin:g the images from the user
    folder = sg.popup_get_folder('Image folder to open', default_path=path)
    if not folder:
        sg.popup_cancel('Cancelling')
        raise SystemExit()

    # PIL supported image types
    img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")

    # get list of files in folder
    flist0 = os.listdir(folder)

    # create sub list of image files (no sub folders, no wrong file types)
    fnames = [f for f in flist0 if os.path.isfile(
        os.path.join(folder, f)) and f.lower().endswith(img_types)]

    num_files = len(fnames)  # number of iamges found
    if num_files == 0:
        sg.popup('No files in folder')
        raise SystemExit()

    del flist0  # no longer needed

    # make these 2 elements outside the layout as we want to "update" them later
    # initialize to the first file in the list
    filename = os.path.join(folder, fnames[0])  # name of first file in list
    image_elem = sg.Image(data=get_img_data(filename, first=True))
    filename_display_elem = sg.Text(filename, size=(80, 3))
    file_num_display_elem = sg.Text('File 1 of {}'.format(num_files), size=(15, 1))

    # define layout, show and read the form
    col = [[filename_display_elem],
           [image_elem]]

    col_files = [[sg.Listbox(values=fnames, change_submits=True, size=(60, 30), key='listbox')],
                 [sg.Button('Next', size=(8, 2)), sg.Button('Prev', size=(8, 2)), file_num_display_elem]]

    layout = [[sg.Column(col_files), sg.Column(col)]]

    window = sg.Window('Image Browser', layout, return_keyboard_events=True,
                       location=(0, 0), use_default_focus=False)

    # loop reading the user input and displaying image, filename
    i = 0
    while True:
        # read the form
        event, values = window.read()
        # print(event, values)
        # perform button and keyboard operations
        if event == sg.WIN_CLOSED:
            break
        elif event in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34'):
            i += 1
            if i >= num_files:
                i -= num_files
            filename = os.path.join(folder, fnames[i])
        elif event in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33'):
            i -= 1
            if i < 0:
                i = num_files + i
            filename = os.path.join(folder, fnames[i])
        elif event == 'listbox':  # something from the listbox
            f = values["listbox"][0]  # selected filename
            filename = os.path.join(folder, f)  # read this file
            i = fnames.index(f)  # update running index
        else:
            filename = os.path.join(folder, fnames[i])

        # update window with new image
        image_elem.update(data=get_img_data(filename, first=True))
        # update window with filename
        filename_display_elem.update(filename)
        # update page display
        file_num_display_elem.update('File {} of {}'.format(i + 1, num_files))

    window.close()
