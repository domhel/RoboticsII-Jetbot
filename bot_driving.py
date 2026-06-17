import time

import cv2
import numpy as np
import onnxruntime as rt
import yaml

from PUTDriver import PUTDriver, gstreamer_pipeline


class AI:
    def __init__(self, config: dict):
        model_cfg = config['model']
        self.path = model_cfg['path']
        self.mode = model_cfg.get('preprocessing', 'color')
        self.swap_rb = bool(model_cfg.get('swap_rb', False))
        self.crop_top = float(model_cfg.get('crop_top', 0.0))

        self.sess = rt.InferenceSession(
            self.path,
            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

        if self.mode == 'grayscale':
            self.mean = np.array([0.449, 0.449, 0.449], dtype=np.float32).reshape(3, 1, 1)
            self.std = np.array([0.226, 0.226, 0.226], dtype=np.float32).reshape(3, 1, 1)
        else:
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Optional top crop (e.g. to drop the horizon / ceiling).
        if self.crop_top > 0.0:
            roi_start = int(img.shape[0] * self.crop_top)
            img = img[roi_start:, :]

        if self.mode == 'grayscale':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)          # (1, H, W)
            img = np.repeat(img, 3, axis=0)            # (3, H, W)
        else:
            # swap_rb: true only for models trained on RGB.
            if self.swap_rb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))         # HWC -> CHW

        img = (img - self.mean) / self.std
        img = np.expand_dims(img, axis=0)              # (1, 3, 224, 224)
        return img

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        detections = np.asarray(detections, dtype=np.float32).reshape(-1)[:2]
        # The regression head is unbounded; clip into the valid command range
        # instead of asserting (a spike must not crash the control loop).
        return np.clip(detections, -1.0, 1.0)

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, 224, 224)

        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)

        return outputs


class SafetyController:
    """Turns raw model outputs into safe, smoothed motor commands.
       (Stuff is implemented but we didn't use most of those actually)
    """

    def __init__(self, config: dict):
        d = config.get('driving', {})
        self.forward_gain = float(d.get('forward_gain', 1.0))
        self.steering_gain = float(d.get('steering_gain', 1.0))
        self.fixed_forward = d.get('fixed_forward', None)
        self.max_forward = float(d.get('max_forward', 1.0))
        self.max_left = float(d.get('max_left', 1.0))
        self.smoothing = float(d.get('smoothing', 0.0))
        self.control_hz = float(d.get('control_hz', 0.0))
        self.turn_slowdown = float(d.get('turn_slowdown', 1.0))

        self._prev_forward = 0.0
        self._prev_left = 0.0
        self._last_time = 0.0

    def _rate_limit(self) -> None:
        if self.control_hz <= 0:
            return
        period = 1.0 / self.control_hz
        now = time.monotonic()
        wait = period - (now - self._last_time)
        if wait > 0:
            time.sleep(wait)
        self._last_time = time.monotonic()

    def __call__(self, model_forward: float, model_left: float):
        # Steering
        left = model_left * self.steering_gain
        left = float(np.clip(left, -self.max_left, self.max_left))

        # Forward
        if self.fixed_forward is not None:
            forward = float(self.fixed_forward)
        else:
            forward = model_forward * self.forward_gain
        # Slow down in sharp turns
        forward *= max(0.0, 1.0 - self.turn_slowdown * abs(left))
        forward = float(np.clip(forward, -self.max_forward, self.max_forward))

        # Exponential moving average
        a = self.smoothing
        forward = a * self._prev_forward + (1 - a) * forward
        left = a * self._prev_left + (1 - a) * left
        self._prev_forward, self._prev_left = forward, left

        self._rate_limit()
        return forward, left


def main():
    with open("config.yml", "r") as stream:
        config = yaml.safe_load(stream)

    driver = PUTDriver(config=config)
    ai = AI(config=config)
    controller = SafetyController(config=config)

    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(flip_method=0, display_width=224, display_height=224),
        cv2.CAP_GSTREAMER,
    )

    ret, image = video_capture.read()
    if not ret:
        print('No camera')
        return
    _ = ai.predict(image)

    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print('No camera')
            break

        model_forward, model_left = ai.predict(image)
        forward, left = controller(model_forward, model_left)


if __name__ == '__main__':
    main()
