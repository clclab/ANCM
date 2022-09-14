from pathlib import Path
import wget
import scipy.io
import numpy as np

class HPfMRILoader:
    def __init__(self, subj_ids):
        self.subj_ids = subj_ids
        self.paths_matfiles = []
        self.data_for_subj = {}

    def download_data(self, files_url, download_dir):
        """
        Download .mat files with fMRI data for each subject into download_dir.
        This function assumes the files are stored at files_url and and named
        subject_<subj_id>.mat, e.g. subject_1.mat, subject_2.mat etc.
        """
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        print(f"Downloading subject .mat files from {files_url} to {download_dir}")
        for subj_id in self.subj_ids:
            filename = f"subject_{subj_id}.mat"
            if not (download_dir / filename).exists():
                print(f"\nsubject {subj_id}")
                wget.download(
                    f"{files_url}{filename}", f"{str(download_dir / filename)}"
                )
            else:
                print(f"File exists: {str(download_dir / filename)}")
            self.paths_matfiles.append(f"{str(download_dir / filename)}")

    def load_subj_dict(self, filepath: str) -> dict:
        """
        Read one .mat file with the data for a single subject as provided by Wehbe et al. (2014)
        into Python dictionary format.

        Description of the original Wehbe data:
        http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/files/description.txt

        Parameters
        ----------
        filepath : str
            Path to the .mat file with data for a single subject, provided by Wehbe et al.

        Returns
        -------
        subj_dict : dict
            Nested dictionary with all information contained in the Wehbe data (defined in the
            meta, words, time and data variables). Additionally, an 'ncols' field is added to
            the 'meta' dict, for cases in which the number of columns in the data does not
            correspond to the number listed by Wehbe et al. as 'nvoxels'.

            The 'meta' subdictionary provides information about the dataset:

            subj_dict['meta']['subject'] = identifier for the human subject (int)
            subj_dict['meta']['nTRs'] = number of rows (/TRs) in the data set (int)
            subj_dict['meta']['ncols'] = number of columns in the data set (int)
            subj_dict['meta']['nvoxels'] = number of voxels (3D pixels) in each image as
            provided by Wehbe et al. Note: this does not always correspond to the number of
            columns in the provided data (int)
            subj_dict['meta']['dimx'] = the maximum x coordinate in the brain image (int)
            subj_dict['meta']['dimy'] = the maximum y coordinate in the brain image (int)
            subj_dict['meta']['dimz'] = the maximum z coordinate in the brain image (int)
            subj_dict['meta']['colToCoord'] = geometric coordinates (x,y,z) of the voxel
            corresponding to each column in the data (numpy array of shape (ncols, 3))
            subj_dict['meta']['coordToCol'] = column indices (within the data) of the
            voxels with geom. coordinates (x,y,z) (numpy array of shape (dimx, dimy, dimz))
            subj_dict['meta']['colToROInum'] = the ROI number according to the AAL atlas
            for each voxel (numpy array of shape (ncols,))
            subj_dict['meta']['coordToROInum'] = ROI number according to the AAL atlas for
            voxels with coordinates (x,y,z) (numpy array of shape (dimx, dimy, dimz))
            subj_dict['meta']['ROInumToName'] = ROI names (subj_dict['meta']['ROInumToName'][i]
            gives the name for number i) (numpy array of shape (117,))
            subj_dict['meta']['voxel_size'] = size of the voxels (3mm*3mm*3mm for all subjects)
            (numpy array of shape (3,))
            subj_dict['meta']['matrix'] = map to the MNI space (numpy array of shape (4,4))

            The 'words' subdictionary describes the sequence of words in the natural story
            shown to the subject:

            subj_dict['words']['text'] = text string as displayed to the subject (numpy array
            of shape (5176,))
            subj_dict['words']['start'] = start time of this text string display measured from
            the beginning of the experiment (numpy array of shape (5176,))
            subj_dict['words']['length'] = the time that the word was on the screen (words
            were displayed for 0.5 seconds each) (numpy array of shape (5176,))

            The subj_dict['time'] matrix is a numpy array of shape (nTRs, 2) in which the
            first column indicates the start time of the recording of each row in the data
            matrix. The second column indicates the run to which every row belongs. The
            experiment consisted of 4 runs, each starting with 20 seconds (=10 TRs) of rest,
            and ending with 10 seconds (=5 TRs) of rest. To be specific, each run ends with
            4.25 TRs of rest, because 3 words appear in the last but 5th TR. These can be
            discarded for convenience/uniformity.

            The subj_dict['data'] matrix contains the raw observed data. The fMRI data is a
            sequence of images collected over time. The data structure is a numpy array of
            shape (nTRs, nvoxels), with one row per image acquired. The element
            subj_dict['data'][t,v] gives the fMRI observation at voxel v during TR t. The full
            image at time t is given by subj_dict['data'][t,:].

            The fMRI images provided by Wehbe et al. have already been realigned, slice timing
            corrected, coregistered with the subject's anatomical scan, normalized to the
            MNI space, and cerebrospinal fluid voxels were removed.
        """

        # load .mat file
        subj_mat = scipy.io.loadmat(filepath)

        # create dict for this subject
        subj_dict = {}

        # prepare subdictionaries
        subj_dict["meta"] = {n: [] for n in subj_mat["meta"].dtype.names}
        subj_dict["words"] = {n: [] for n in subj_mat["words"].dtype.names}

        # extract data & fill dicts
        subj_dict["time"] = subj_mat["time"].squeeze()
        subj_dict["data"] = subj_mat["data"].squeeze()
        subj_dict["meta"]["subject"] = int(subj_mat["meta"]["subject"].squeeze())
        subj_dict["meta"]["nTRs"] = int(subj_mat["meta"]["nTRs"].squeeze())
        subj_dict["meta"]["nvoxels"] = int(subj_mat["meta"]["nvoxels"].squeeze())
        subj_dict["meta"]["ncols"] = subj_dict["data"].shape[1]
        subj_dict["meta"]["dimx"] = int(subj_mat["meta"]["dimx"].squeeze())
        subj_dict["meta"]["dimy"] = int(subj_mat["meta"]["dimy"].squeeze())
        subj_dict["meta"]["dimz"] = int(subj_mat["meta"]["dimz"].squeeze())
        subj_dict["meta"]["colToCoord"] = subj_mat["meta"]["colToCoord"].squeeze()[()]
        subj_dict["meta"]["coordToCol"] = subj_mat["meta"]["coordToCol"].squeeze()[()]
        subj_dict["meta"]["colToROInum"] = (
            subj_mat["meta"]["colToROInum"].squeeze()[()].flatten()
        )
        subj_dict["meta"]["coordToROInum"] = subj_mat["meta"][
            "coordToROInum"
        ].squeeze()[()]
        subj_dict["meta"]["ROInumToName"] = np.array(
            [
                t.squeeze()[()]
                for t in subj_mat["meta"]["ROInumToName"].squeeze()[()].flatten()
            ]
        )
        subj_dict["meta"]["voxel_size"] = (
            subj_mat["meta"]["voxel_size"].squeeze()[()].flatten()
        )
        subj_dict["meta"]["matrix"] = subj_mat["meta"]["matrix"].squeeze()[()]
        subj_dict["words"]["text"] = np.array(
            [t.squeeze()[()].item() for t in subj_mat["words"]["text"].flatten()]
        )
        subj_dict["words"]["start"] = np.array(
            [t.squeeze()[()].item() for t in subj_mat["words"]["start"].flatten()]
        )
        subj_dict["words"]["length"] = np.array(
            [t.squeeze()[()].item() for t in subj_mat["words"]["length"].flatten()]
        )

        # corrections for zero indexing
        # replace 0s in the coordToCol array (referring to deleted voxels) by None
        subj_dict["meta"]["coordToCol"] = np.where(
            subj_dict["meta"]["coordToCol"] == 0,
            None,
            subj_dict["meta"]["coordToCol"] - 1,
        )
        # put 'Not_labelled' at position 0 in the ROInumToName array
        subj_dict["meta"]["ROInumToName"] = np.insert(
            subj_dict["meta"]["ROInumToName"][:-1],
            0,
            subj_dict["meta"]["ROInumToName"][-1],
        )

        return subj_dict

    def read_subj_files(self) -> dict:
        """
        Stores data from all .mat files in the provided list to a single combined dictionary,
        indexed by subject id.

        Parameters
        ----------
        paths_matfiles : list
            List of paths to the .mat files with data per subject, as provided by Wehbe et al.

        Returns
        -------
        data_for_subj : dict
            Combined dict of nested dictionaries where
            data_for_subj[subj_id] = raw data dictionary for each subject
            (as described in the load_subj_dict docstring).
        """

        # match ids & files
        ids_files = zip(self.subj_ids, self.paths_matfiles)

        # store all subject data into one dict
        self.data_for_subj = {
            subj_id: self.load_subj_dict(matfile) for subj_id, matfile in ids_files
        }

    def save_subj_dicts(self, interim_dir: str, filename: str):
        Path(interim_dir).mkdir(parents=True, exist_ok=True)
        filepath = str(interim_dir / filename)
        np.save(filepath, self.data_for_subj)
        print(f"Saved raw fMRI data dict to {filepath}")


def text_between(start_time: float, end_time: float, subj_dict: dict) -> list:
    """
    Retrieves the text presented to subjects between two timepoints given in seconds since
    start of the experiment (up to but excluding end_time). It returns the items from the
    text array in the 'words' subdictionary, for which the start times fall between the two
    given timepoints.

    Parameters
    ----------
    start_time : float
        Beginning of the time interval (seconds since start of the experiment).
    end_time : float
        End of the time interval (seconds since start of the experiment), the interval does
        not include this point.
    subj_dict : dict
        Data dictionary for a single subject (as described in the load_subj_dict docstring).

    Returns
    -------
    text : list
        List of words presented between the two timepoints.
    """
    words = subj_dict["words"]
    idx = np.argwhere((words["start"] >= start_time) & (words["start"] < end_time))
    text = [w[0] for w in words["text"][idx]]
    return text


def text_at_tr(tr: int, subj_dict: dict) -> list:
    """
    Retrieves the text presented to the subject at the given TR.

    Parameters
    ----------
    tr : int
        Index of the TR (between 0 and nTRs) for which to retrieve the presented text.
    subj_dict : dict
        Data dictionary for a single subject (as described in the load_subj_dict docstring).

    Returns
    -------
    text : list
        List of words (sequentially) presented during this TR.
    """
    time = subj_dict["time"]

    final_tr = len(time) - 1
    if tr < final_tr:
        tr_end = time[tr + 1, 0]
    elif tr == final_tr:
        tr_end = time[tr, 0]
    else:
        return ValueError("TR {} does not exist, final TR is {}.".format(tr, final_tr))

    tr_start = time[tr, 0]
    text = text_between(tr_start, tr_end, subj_dict)
    return text


def get_tr_texts(subj_dict: dict):
    """
    Retrieves the text presented to the subject at the given TR.

    Parameters
    ----------
    subj_dict : dict
        Data dictionary for a single subject (as described in the load_subj_dict docstring).

    Returns
    -------
    tr_texts : list
        List of strings of text presented at each text TR.
    """
    nTRs = subj_dict["meta"]["TRs"]
    tr_texts = [" ".join(text_at_tr(tr, subj_dict)) for tr in range(nTRs)[:-1]]
    return tr_texts
