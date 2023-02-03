import ffmpeg
from pathlib import Path
from params import Params


def extract_frames_from_video(
    input_file, output_dir, output_ext=Params.image_extension
):
    input_file_path = Path(input_file)
    output_path = Path(output_dir)

    job = ffmpeg.input(str(input_file_path))
    kwargs = {"pix_fmt": "rgb24"}

    if output_ext == ".jpg":
        kwargs.update({"q:v": "2"})  # highest quality for jpg

    job = job.output(str(output_path / ("%5d." + output_ext)), **kwargs)

    try:
        job.run()
    except Exception as e:
        print(e)
