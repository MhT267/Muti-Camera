import os
import PySpin
import sys
import platform
import numpy as np
import cv2


class StreamMode:
    """
    'Enum' for choosing stream mode
    """
    STREAM_MODE_TELEDYNE_GIGE_VISION = 0  # Teledyne Gige Vision stream mode is the default stream mode for spinview which is supported on Windows
    STREAM_MODE_PGRLWF = 1  # Light Weight Filter driver is our legacy driver which is supported on Windows
    STREAM_MODE_SOCKET = 2  # Socket is supported for MacOS and Linux, and uses native OS network sockets instead of a filter driver


# Determine stream mode based on current OS
system = platform.system()
# Print out OS based on result
if system == "Windows":
    print("Using Stream mode STREAM_MODE_TELEDYNE_GIGE_VISION")
    CHOSEN_STREAMMODE = StreamMode.STREAM_MODE_TELEDYNE_GIGE_VISION
elif system == "Linux" or system == "Darwin":
    print("Using Stream mode STREAM_MODE_SOCKET")
    CHOSEN_STREAMMODE = StreamMode.STREAM_MODE_SOCKET
else:
    print("OS Unknown; Using Stream mode STREAM_MODE_SOCKET")
    CHOSEN_STREAMMODE = StreamMode.STREAM_MODE_SOCKET

NUM_IMAGES = 10  # number of images to grab


def set_stream_mode(cam):
    """
    This function changes the stream mode

    :param cam: Camera to change stream mode.
    :type cam: CameraPtr
    :type nodemap_tlstream: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    streamMode = "Socket"

    if CHOSEN_STREAMMODE == StreamMode.STREAM_MODE_TELEDYNE_GIGE_VISION:
        streamMode = "TeledyneGigeVision"
    elif CHOSEN_STREAMMODE == StreamMode.STREAM_MODE_PGRLWF:
        streamMode = "LWF"
    elif CHOSEN_STREAMMODE == StreamMode.STREAM_MODE_SOCKET:
        streamMode = "Socket"

    result = True

    # Retrieve Stream nodemap
    nodemap_tlstream = cam.GetTLStreamNodeMap()

    # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
    node_stream_mode = PySpin.CEnumerationPtr(nodemap_tlstream.GetNode('StreamMode'))

    # The node "StreamMode" is only available for GEV cameras.
    # Skip setting stream mode if the node is inaccessible.
    if not PySpin.IsReadable(node_stream_mode) or not PySpin.IsWritable(node_stream_mode):
        return True

    # Retrieve the desired entry node from the enumeration node
    node_stream_mode_custom = PySpin.CEnumEntryPtr(node_stream_mode.GetEntryByName(streamMode))

    if not PySpin.IsReadable(node_stream_mode_custom):
        # Failed to get custom stream node
        print('Stream mode ' + streamMode + ' not available. Aborting...')
        return False

    # Retrieve integer value from entry node
    stream_mode_custom = node_stream_mode_custom.GetValue()

    # Set integer as new value for enumeration node
    node_stream_mode.SetIntValue(stream_mode_custom)

    print('Stream Mode set to %s...' % node_stream_mode.GetCurrentEntry().GetSymbolic())
    return result


def acquire_images(cam, nodemap, nodemap_tldevice):
    try:
        result = True
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        cam.BeginAcquisition()

        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        while True:  # 持续采集直到按下q键
            image_result = cam.GetNextImage(1000)
            if image_result.IsIncomplete():
                continue

            # 转换为OpenCV格式
            image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
            image_data = image_converted.GetData()
            height, width = image_converted.GetHeight(), image_converted.GetWidth()
            cv_image = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 3)

            # 显示图像
            cv2.imshow('FLIR Camera', cv_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            image_result.Release()

        cam.EndAcquisition()
        cv2.destroyAllWindows()
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not readable.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Set Stream Modes
        result &= set_stream_mode(cam)

        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """

    # Since this application saves images in the current folder
    # we must ensure that we have permission to write to this folder.
    # If we do not have permission, fail right away.
    try:
        test_file = open('test.txt', 'w+')
    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):
        print('Running example for camera %d...' % i)

        result &= run_single_camera(cam)
        print('Camera %d example complete... \n' % i)

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
