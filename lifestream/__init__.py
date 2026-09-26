"""LifeStream AI — real-time pool drowning-detection with Computer Vision.

Pipeline: camera/scene -> SwimDetector (pose per swimmer) -> CentroidTracker
(per-swimmer timelines) -> SwimFeatureExtractor (posture/limb/submersion
features) -> DrowningRiskEngine (scored risk + phases) -> Alerts + HUD.
"""

__version__ = "0.1.0"
__app_name__ = "LifeStream AI"
