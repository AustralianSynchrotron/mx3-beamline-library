import typing
import pytest
import imageio
import numpy as np

if typing.TYPE_CHECKING:
    from mx3_beamline_library.devices.sim.classes.detectors import SimBlackFlyCam


@pytest.mark.parametrize(
    "camera",
    (
        "sim_blackfly_camera",
    ),
    indirect=True,
)
class TestCameras:
    """Run camera tests."""

    def test_camera_setup(self, camera: "SimBlackFlyCam"):
        """Test camera device initialised.

        Parameters
        ----------
        camera : SimBlackFlyCam
            Camera device instance.
        """

        assert camera is not None

    @pytest.mark.parametrize(
        (
            "image_dir", "width", "height", "depth", "as_gray",
        ),
        (
            ("./tests/test_images/snapshot_6.jpg", 2048, 1536, 3, False),
            ("./tests/test_images/snapshot_6.jpg", 2048, 1536, 0, True),
        ),
    )
    def test_set_values(
        self,
        camera: "SimBlackFlyCam",
        image_dir: str,
        width: int,
        height: int,
        depth: int,
        as_gray: bool,
    ):
        """Test camera device "set_values" method.

        Parameters
        ----------
        camera : SimBlackFlyCam
            Camera device instance.
        """

        # Clear device signals
        camera.array_data.set(None)
        camera.width.set(None)
        camera.height.set(None)
        camera.depth.set(None)

        # Attempt to set all signals by reading in a Numpy array from an image file
        snapshot = imageio.v2.imread(image_dir, as_gray=as_gray)
        camera.set_values(snapshot)

        # Check that values were read into the signals correctly
        assert snapshot is not None
        np.testing.assert_array_equal(snapshot, camera.array_data.get())
        assert camera.width.get() == width
        assert camera.height.get() == height
        assert camera.depth.get() == depth
