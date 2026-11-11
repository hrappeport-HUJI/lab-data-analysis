from scanlag.cli import split_prompt_args


def test_split_prompt_args():
    assert split_prompt_args([r"D:\My scans\exp --redo --detector classical"]) == [
        r"D:\My scans\exp", "--redo", "--detector", "classical"]
    assert split_prompt_args(['D:\\exp\\" --redo']) == ["D:\\exp\\", "--redo"]
    assert split_prompt_args([r"D:\exp", "--redo"]) == [r"D:\exp", "--redo"]
    assert split_prompt_args([r'"D:\My scans\exp" --out "D:\res ults"']) == [
        r"D:\My scans\exp", "--out", r"D:\res ults"]
