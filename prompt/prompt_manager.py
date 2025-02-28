from pathlib import Path

class PromptManager:
    def __init__(self, prompt_dir="./prompt/prompts"):
        self.prompt_dir = Path(prompt_dir)
        self.prompt_files = sorted(self.prompt_dir.iterdir(), key=lambda x: x.name)

        file_template_map = {
            "yaw_vqa_template": self.prompt_files[0],
            "xy2d_vqa_template": self.prompt_files[1],
            "depth_vqa_template": self.prompt_files[2],
            "dis_vqa_template": self.prompt_files[3],
            "lr_vqa_template": self.prompt_files[4],
            "fb_vqa_template": self.prompt_files[5]
        }
        for name, file_path in file_template_map.items():
            with open(file_path) as f:
                setattr(self, name, f.read())
  
        self.start_marker = "<OUTPUT FORMAT START>"
        self.end_marker = "<OUTPUT FORMAT END>"
        self.markers = [self.start_marker, self.end_marker]
        self.pattern = self.start_marker + r"(.*?)" + self.end_marker

