from setuptools import find_packages, setup

package_name = "orbbec_dual_sync"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/dual_sync_detect.launch.py"]),
        ("share/" + package_name + "/config", ["config/dual_sync.yaml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Jessica Camera",
    maintainer_email="devnull@example.com",
    description="Dual Orbbec publishers + dual-view rack/arm overlay",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "orbbec_camera_publisher = orbbec_dual_sync.camera_publisher:main",
            "orbbec_sync_forward = orbbec_dual_sync.sync_forward_node:main",
            "orbbec_rack_overlay = orbbec_dual_sync.rack_overlay_node:main",
            "orbbec_dual_view = orbbec_dual_sync.dual_view_node:main",
        ],
    },
)
