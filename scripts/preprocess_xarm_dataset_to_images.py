import concurrent.futures
import dataclasses
import io
import json
import pathlib
import shutil
import typing

import av
import datasets
import pyarrow.parquet as pq
import tyro


@dataclasses.dataclass
class Args:
    src_root: str = "data/xarm_data_heyao_2"
    dst_root: str = "data/xarm_data_heyao_2_images"
    image_keys: tuple[str, ...] = ("observation.images.front", "observation.images.gripper_color")
    image_ext: str = "png"
    storage_mode: typing.Literal["paths", "embedded"] = "paths"
    embedded_format: typing.Literal["png", "jpeg"] = "jpeg"
    jpeg_quality: int = 95
    max_workers: int = 8
    overwrite: bool = False


def _decode_video_to_images(video_path: pathlib.Path, output_dir: pathlib.Path, image_ext: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            image = frame.to_image()
            image_path = output_dir / f"{frame_index:06d}.{image_ext}"
            image.save(image_path)
            paths.append(str(image_path))
    return paths


def _decode_video_to_embedded_images(
    video_path: pathlib.Path,
    image_format: typing.Literal["png", "jpeg"],
    jpeg_quality: int,
) -> list[dict[str, bytes | None]]:
    images: list[dict[str, bytes | None]] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            image = frame.to_image()
            buffer = io.BytesIO()
            save_kwargs = {}
            save_format = image_format.upper()
            if image_format == "jpeg":
                save_format = "JPEG"
                save_kwargs["quality"] = jpeg_quality
                save_kwargs["optimize"] = True
            image.save(buffer, format=save_format, **save_kwargs)
            images.append({"bytes": buffer.getvalue(), "path": None})
    return images


def _build_episode_with_images(
    src_root: pathlib.Path,
    dst_root: pathlib.Path,
    episode_path: pathlib.Path,
    video_path_template: str,
    image_keys: tuple[str, ...],
    image_ext: str,
    storage_mode: typing.Literal["paths", "embedded"],
    embedded_format: typing.Literal["png", "jpeg"],
    jpeg_quality: int,
) -> None:
    table = pq.read_table(episode_path)
    data = table.to_pydict()
    episode_index = int(data["episode_index"][0])
    episode_chunk = episode_index // 1000

    for image_key in image_keys:
        rel_video_path = video_path_template.format(
            episode_chunk=episode_chunk,
            video_key=image_key,
            episode_index=episode_index,
        )
        src_video_path = src_root / rel_video_path
        if storage_mode == "paths":
            dst_image_dir = (
                dst_root / "images" / f"chunk-{episode_chunk:03d}" / image_key / f"episode_{episode_index:06d}"
            )
            images = _decode_video_to_images(src_video_path, dst_image_dir, image_ext)
        else:
            images = _decode_video_to_embedded_images(src_video_path, embedded_format, jpeg_quality)

        if len(images) != len(data["frame_index"]):
            raise ValueError(
                f"Frame count mismatch for episode {episode_index} key {image_key}: "
                f"{len(images)} decoded vs {len(data['frame_index'])} parquet rows"
            )
        data[image_key] = images

    features = datasets.Features(
        {
            "action": datasets.Sequence(datasets.Value("float32"), length=7),
            "observation.state": datasets.Sequence(datasets.Value("float32"), length=7),
            "observation.images.front": datasets.Image(),
            "observation.images.gripper_color": datasets.Image(),
            "timestamp": datasets.Value("float32"),
            "frame_index": datasets.Value("int64"),
            "episode_index": datasets.Value("int64"),
            "index": datasets.Value("int64"),
            "task_index": datasets.Value("int64"),
        }
    )

    episode_dataset = datasets.Dataset.from_dict(data, features=features, split="train")
    dst_episode_path = dst_root / "data" / episode_path.relative_to(src_root / "data")
    dst_episode_path.parent.mkdir(parents=True, exist_ok=True)
    episode_dataset.to_parquet(str(dst_episode_path))


def main(args: Args) -> None:
    src_root = pathlib.Path(args.src_root)
    dst_root = pathlib.Path(args.dst_root)

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_root} already exists, pass --overwrite to replace it.")
        shutil.rmtree(dst_root)

    shutil.copytree(src_root / "meta", dst_root / "meta")

    info_path = dst_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = None
    info["total_videos"] = 0
    info["features"] = {
        "action": info["features"]["action"],
        "observation.state": info["features"]["observation.state"],
        "observation.images.front": {
            "dtype": "image",
            "shape": info["features"]["observation.images.front"]["shape"],
            "names": info["features"]["observation.images.front"]["names"],
        },
        "observation.images.gripper_color": {
            "dtype": "image",
            "shape": info["features"]["observation.images.gripper_color"]["shape"],
            "names": info["features"]["observation.images.gripper_color"]["names"],
        },
        "timestamp": info["features"]["timestamp"],
        "frame_index": info["features"]["frame_index"],
        "episode_index": info["features"]["episode_index"],
        "index": info["features"]["index"],
        "task_index": info["features"]["task_index"],
    }
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False))

    episode_paths = sorted((src_root / "data").rglob("episode_*.parquet"))
    video_path_template = json.loads((src_root / "meta" / "info.json").read_text())["video_path"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                _build_episode_with_images,
                src_root,
                dst_root,
                episode_path,
                video_path_template,
                args.image_keys,
                args.image_ext,
                args.storage_mode,
                args.embedded_format,
                args.jpeg_quality,
            )
            for episode_path in episode_paths
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main(tyro.cli(Args))
