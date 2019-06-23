import turicreate as tc
import sys

def explore(sframe_dir=None, draw_bounding_boxes=True, range = (0,20)):
    """
    Allows the user to explore by inspection the created SFrame.

    Parameters
    ----------
    sframe_dir : str, optional
        The path of the SFrame file.
    draw_bounding_boxes : bool, optional
        Whether or not to display the images with the bounding boxes drawn (good for manual inspection).
    limit : int, optional
        How many entries to display in the Turi Create Visualizer.
    """
    sframe_dir = sframe_dir.strip()
    sf = tc.SFrame(sframe_dir)
    sf.dropna()
    print( sf[range[0]: range[1]])
    if draw_bounding_boxes:
        sf['image_with_ground_truth'] = tc.object_detector.util.draw_bounding_boxes(sf['image'], sf['annotations'])
        sf['image_with_ground_truth'].apply(lambda x: x)
    sf[range[0]: range[1]].explore()


if __name__ == "__main__":
    explore(sys.argv[1], range = (0,10))