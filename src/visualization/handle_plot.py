from datetime import datetime
from os import makedirs
from os.path import join as path_join


def save_plot(fig, plt, file_path, file_name):
    fig.tight_layout()

    makedirs(file_path, exist_ok=True)
    plt.savefig(path_join(file_path, f'{file_name} - {datetime.now().strftime("%H-%M-%S %d-%m-%Y")}.png'),
                bbox_inches='tight')
    plt.close(fig)
