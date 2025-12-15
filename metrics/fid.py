import os.path as osp
import os
from typing import List

import joblib
import librosa
import numpy as np
from scipy import linalg
import audio as Audio
from fastdtw import fastdtw

import torchaudio as ta
from meldataset import mel_spectrogram

from tqdm import tqdm

class CalFeature:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.all_feature_type = ["mfcc", "mel", "mfcc_un_norm"]
        self.feature_type2dim_dict = dict(
            mfcc=20, mel=80, mfcc_un_norm=20,
        )
        self.cal_type = ""  # This attribute needs to be overridden in the subclass


    def compute_mfcc(self, wav_filepath):
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=librosa.load(wav_filepath)[0], sr=self.sample_rate).T  # (seq_len,20)

        # Normalize the aligned MFCC features
        return mfcc / np.linalg.norm(mfcc, axis=0)  # (seq_len,20)

    def compute_mfcc_un_norm(self, wav_filepath):
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=librosa.load(wav_filepath)[0], sr=self.sample_rate).T  # (seq_len,20)
        return mfcc  # (seq_len,20)

    def compute_mel(self, wav_filepath):
        """
        This should not implement the cache method here, but due to environmental issues, it is implemented here.
        :param wav_filepath:
        :return:
        """

        audio, sr = ta.load(wav_filepath)
        mel = mel_spectrogram(
            y=audio, 
            n_fft=1024, 
            num_mels=80, 
            sampling_rate=sr, 
            hop_size=256,
            win_size=1024,
            fmin=0, 
            fmax=8000, 
            center=False
        ).squeeze()
        
        mel = mel.numpy() #[80, T]

        return mel.T # (T,80)
    
    @staticmethod
    def manifold_estimate(A_features, B_features, k):
        """
        A partially parallel computation method that reduces memory overhead, suitable for most cases.
        If you are not satisfied with the computation speed and still have memory available, you can adjust the splitting method to trade memory for computation time.
        :param A_features:
        :param B_features:
        :param k:
        :return:
        """
        a_len_sqrt = int(np.sqrt(len(A_features)))
        kth_smallest_dis_list = list()
        for i, j in zip(range(a_len_sqrt), range(a_len_sqrt-1, -1, -1)):
            if j != 0:
                mini_distances_a = np.linalg.norm(
                    A_features[i * a_len_sqrt:(i + 1) * a_len_sqrt, np.newaxis, :] - A_features,
                    axis=2)
            else:
                mini_distances_a = np.linalg.norm(
                    A_features[i * a_len_sqrt:, np.newaxis, :] - A_features,
                    axis=2)
            mini_kth_smallest_dis = np.partition(mini_distances_a, k, axis=1)[:, k]
            del mini_distances_a
            kth_smallest_dis_list.append(mini_kth_smallest_dis)
        kth_smallest_dis = np.concatenate(kth_smallest_dis_list)
        count = 0
        b_len_sqrt = int(np.sqrt(len(B_features)))
        for i in range(a_len_sqrt+1):
            mini_distances_b2a = np.linalg.norm(
                B_features[i * b_len_sqrt:(i + 1) * b_len_sqrt, np.newaxis, :] - A_features, axis=2)
            mini_result = mini_distances_b2a - kth_smallest_dis[np.newaxis, :]
            count += int(np.sum(np.any(mini_result <= 0, axis=1)))
        return count / len(B_features)

    @staticmethod
    def manifold_estimate_fully_parallel(A_features, B_features, k):
        """
        This implementation is fully parallel and will put enormous pressure on memory. Use with caution.
        You should monitor memory usage while running. To check memory usage, use: free -h
        Use feature A to cover feature B.
        :param A_features: np.ndarray(n,feature_dim)
        :param B_features: np.ndarray(n,feature_dim)
        :param k:
        :return:
        """
        distances_a = np.linalg.norm(A_features[:, np.newaxis, :] - A_features, axis=2)
        fourth_smallest = np.partition(distances_a, k, axis=1)[:, k]
        del distances_a
        distances_b2a = np.linalg.norm(B_features[:, np.newaxis, :] - A_features, axis=2)
        result = distances_b2a - fourth_smallest[np.newaxis, :]
        del distances_b2a
        del fourth_smallest
        count = np.sum(np.any(result <= 0, axis=1))
        return count / len(B_features)

    @staticmethod
    def manifold_estimate_fully_sequential(A_features, B_features, k):
        """
        The current scheme is fully sequential and runs relatively slowly. Use with caution.
        :param A_features:
        :param B_features:
        :param k:
        :return:
        """
        KNN_list_in_A = {}
        for i, A in enumerate(A_features):
            pairwise_distances = np.zeros(shape=(len(A_features)))

            for j, A_prime in enumerate(A_features):
                d = np.linalg.norm((A - A_prime), ord=2)
                pairwise_distances[j] = d

            v = np.partition(pairwise_distances, k)[k]
            KNN_list_in_A[i] = v

        n = 0

        for B in B_features:
            for i, A_prime in enumerate(A_features):
                d = np.linalg.norm((B - A_prime), ord=2)
                if d <= KNN_list_in_A[i]:
                    n += 1
                    break  # Once a cover is found, the search for the next one stops.

        return n / len(B_features)
    
    @staticmethod
    def manifold_estimate_faiss(A_features, B_features, k):
        # """
        # This implementation uses FAISS for acceleration. It requires FAISS to be installed.
        # :param A_features:
        # :param B_features:
        # :param k:
        # :return:
        # """
        # import faiss

        # d = A_features.shape[1]  # Dimension of the feature
        # index = faiss.IndexFlatL2(d)  # Build the index
        # index.add(A_features.astype(np.float32))  # Add A features to the index

        # # Search for the k+1 nearest neighbors in A for each feature in A to get the k-th distance
        # D_A, I_A = index.search(A_features.astype(np.float32), k + 1)
        # kth_distances = D_A[:, -1]  # The k-th nearest distance for each feature in A

        # # Now search for the nearest neighbors in A for each feature in B
        # D_B, I_B = index.search(B_features.astype(np.float32), 1)

        # count = 0
        # for i in range(B_features.shape[0]):
        #     if np.any(D_B[i] <= kth_distances):
        #         count += 1

        # return count / B_features.shape[0]
        """
        Manifold coverage estimation using k-NN (Faiss).

        Parameters
        ----------
        A_features : np.ndarray, shape (N_A, D)
            Reference feature set (defines the manifold).
        B_features : np.ndarray, shape (N_B, D)
            Query feature set (to be covered).
        k : int
            k-th nearest neighbor for radius estimation.

        Returns
        -------
        float
            Coverage ratio of B by A.
        """
        import faiss
        # Ensure float32 (Faiss requirement)
        A = np.ascontiguousarray(A_features, dtype=np.float32)
        B = np.ascontiguousarray(B_features, dtype=np.float32)

        dim = A.shape[1]

        # --------------------------------------------------
        # 1. Build index on A
        # --------------------------------------------------
        index = faiss.IndexFlatL2(dim)   # exact L2 k-NN
        index.add(A)

        # --------------------------------------------------
        # 2. Compute k-NN radius for each point in A
        #    (k+1 because nearest neighbor is itself)
        # --------------------------------------------------
        distances_A, _ = index.search(A, k + 1)

        # distances are squared L2
        # radius r_i = sqrt(dist to k-th neighbor)
        radii = np.sqrt(distances_A[:, k])   # shape: (N_A,)

        # --------------------------------------------------
        # 3. For each b_j, find nearest a_i
        # --------------------------------------------------
        distances_B, indices_B = index.search(B, 1)

        # nearest distance
        d_ba = np.sqrt(distances_B[:, 0])     # shape: (N_B,)
        nearest_a_idx = indices_B[:, 0]

        # --------------------------------------------------
        # 4. Check manifold condition
        #    dist(b_j, a_i) <= r_i
        # --------------------------------------------------
        covered = d_ba <= radii[nearest_a_idx]

        return float(np.mean(covered))


class CalFidSeries(CalFeature):
    # This defines how to calculate TTS FID-like functions.
    # Mainly defines how to obtain various features and finally use the same calculation logic.
    def __init__(self, speakers_to_synth_wavs_and_reference, sample_rate=22050):
        """

        @param speakers_to_synth_wavs_and_reference: Dictionary mapping speaker IDs to their synthesized and reference WAV files.
        """
        super().__init__(sample_rate=sample_rate)
        self.cal_type = "fid"
        self.SAMPLING_RATE = sample_rate
        self.speakers_to_synth_wavs_and_reference = speakers_to_synth_wavs_and_reference

    def __call__(self, feature_type):
        assert feature_type in self.all_feature_type
        synth_wav_list = list()
        ref_wav_list = list()
        for speaker_id, wav_dict_list in self.speakers_to_synth_wavs_and_reference.items():
            for wav_dict in wav_dict_list:
                synth_wav_list.append(wav_dict["synthesized_wav"])
                ref_wav_list.append(wav_dict["reference_wav"])

        synth_mean, synth_cov = self.get_f_mean_cov(file_list=synth_wav_list, feature_type=feature_type)
        ref_mean, ref_cov = self.get_f_mean_cov(file_list=ref_wav_list, feature_type=feature_type)
        fid_value = self.__calculate_frechet_distance(synth_mean, synth_cov, ref_mean, ref_cov)
        return fid_value
    

    def get_f_mean_cov(self, file_list: List[str], feature_type: str):
        feature_extractor = None
        if feature_type == "mfcc":
            feature_extractor = self.compute_mfcc
        elif feature_type == "mel":
            feature_extractor = self.compute_mel
        elif feature_type == "mfcc_un_norm":
            feature_extractor = self.compute_mfcc_un_norm
        else:
            raise NotImplementedError(f"Feature type {feature_type} not implemented.")

        features = []
        for file in tqdm(file_list):
            feat = feature_extractor(file)
            features.append(feat)
        
        all_features = np.concatenate(features, axis=0)  # (total_frames, feature_dim)
        mean = np.mean(all_features, axis=0)
        cov = np.cov(all_features, rowvar=False)

        return mean, cov
    

    @staticmethod
    def __calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        This function is the shared part calculated at the end.
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
    

class CalFIDAlign(CalFidSeries):
    # This defines how to calculate TTS FID-like functions with DTW alignment.
    # Mainly defines how to obtain various features and finally use the same calculation logic.
    def __init__(self, speakers_to_synth_wavs_and_reference, sample_rate=22050):
        """

        @param speakers_to_synth_wavs_and_reference: Dictionary mapping speaker IDs to their synthesized and reference WAV files.
        """
        super().__init__(sample_rate=sample_rate)
        self.cal_type = "fid_dtw"
        self.SAMPLING_RATE = sample_rate
        self.speakers_to_synth_wavs_and_reference = speakers_to_synth_wavs_and_reference

    def __call__(self, feature_type, norm=False):
        assert feature_type in self.all_feature_type

        synth_features_list = list()
        ref_features_list = list()
        for speaker_id, wav_dict_list in self.speakers_to_synth_wavs_and_reference.items():
            for wav_dict in wav_dict_list:
                synth_wav = wav_dict["synthesized_wav"]
                ref_wav = wav_dict["reference_wav"]
                aligned_synth_features, aligned_ref_features = self.get_pair_mfcc(
                    synth_wav=synth_wav, 
                    ref_wav=ref_wav, 
                    feature_type=feature_type,
                    norm=norm
                )
                synth_features_list.append(aligned_synth_features)
                ref_features_list.append(aligned_ref_features)


        synth_all_features = np.concatenate(synth_features_list, axis=0)  # (total_frames, feature_dim)
        ref_all_features = np.concatenate(ref_features_list, axis=0)  # (total_frames, feature_dim)

        synth_mean = np.mean(synth_all_features, axis=0)
        synth_cov = np.cov(synth_all_features, rowvar=False)

        ref_mean = np.mean(ref_all_features, axis=0)
        ref_cov = np.cov(ref_all_features, rowvar=False)

        fid_value = self.__calculate_frechet_distance(synth_mean, synth_cov, ref_mean, ref_cov)
        return fid_value

    def get_pair_mfcc(self, synth_wav, ref_wav, feature_type, norm=False):
        deal_fun = getattr(self, "compute_" + feature_type)
        syn_features = deal_fun(synth_wav).T  # (feature_dim, T1)
        ref_features = deal_fun(ref_wav).T  # (feature_dim, T2)

        # Use fastdtw to align two MFCC feature matrices
        _, path = fastdtw(syn_features.T, ref_features.T)
        # Aligned feature matrices
        aligned_syn_features = syn_features[:, [p[0] for p in path]].T
        aligned_ref_features = ref_features[:, [p[1] for p in path]].T
        if norm:
            # Normalize the aligned MFCC features
            aligned_syn_features = aligned_syn_features / np.linalg.norm(aligned_syn_features, axis=0)
            aligned_ref_features = aligned_ref_features / np.linalg.norm(aligned_ref_features, axis=0)
        return aligned_syn_features, aligned_ref_features
    

class CalRecall(CalFeature):
    def __init__(self, speakers_to_synth_wavs_and_reference, sample_rate=22050, k=3):
        """

        @param speakers_to_synth_wavs_and_reference: Dictionary mapping speaker IDs to their synthesized and reference WAV files.
        """
        super().__init__(sample_rate=sample_rate)
        self.cal_type = "recall"
        self.SAMPLING_RATE = sample_rate
        self.speakers_to_synth_wavs_and_reference = speakers_to_synth_wavs_and_reference
        self.k = k

    
    def get_feature(self, file_list: List[str], feature_type: str):
        feature_extractor = None
        if feature_type == "mfcc":
            feature_extractor = self.compute_mfcc
        elif feature_type == "mel":
            feature_extractor = self.compute_mel
        elif feature_type == "mfcc_un_norm":
            feature_extractor = self.compute_mfcc_un_norm
        else:
            raise NotImplementedError(f"Feature type {feature_type} not implemented.")

        features = []
        for file in tqdm(file_list):
            feat = feature_extractor(file)
            features.append(feat)
        
        all_features = np.concatenate(features, axis=0)  # (total_frames, feature_dim)
        return all_features

    def get_recall(self, feature_type: str):
        assert feature_type in self.all_feature_type
        synth_wav_list = list()
        ref_wav_list = list()
        for speaker_id, wav_dict_list in self.speakers_to_synth_wavs_and_reference.items():
            for wav_dict in wav_dict_list:
                synth_wav_list.append(wav_dict["synthesized_wav"])
                ref_wav_list.append(wav_dict["reference_wav"])

        synth_features = self.get_feature(file_list=synth_wav_list, feature_type=feature_type)
        ref_features = self.get_feature(file_list=ref_wav_list, feature_type=feature_type)

        recall_value = self.manifold_estimate_faiss(
            A_features=ref_features,
            B_features=synth_features,
            k=self.k
        )
        return recall_value
    
    def __call__(self, feature_type):
        return self.get_recall(feature_type=feature_type)
    

class CalPrecision(CalFeature):
    def __init__(self, speakers_to_synth_wavs_and_reference, sample_rate=22050, k=3):
        """

        @param speakers_to_synth_wavs_and_reference: Dictionary mapping speaker IDs to their synthesized and reference WAV files.
        """
        super().__init__(sample_rate=sample_rate)
        self.cal_type = "precision"
        self.SAMPLING_RATE = sample_rate
        self.speakers_to_synth_wavs_and_reference = speakers_to_synth_wavs_and_reference
        self.k = k

    
    def get_feature(self, file_list: List[str], feature_type: str):
        feature_extractor = None
        if feature_type == "mfcc":
            feature_extractor = self.compute_mfcc
        elif feature_type == "mel":
            feature_extractor = self.compute_mel
        elif feature_type == "mfcc_un_norm":
            feature_extractor = self.compute_mfcc_un_norm
        else:
            raise NotImplementedError(f"Feature type {feature_type} not implemented.")

        features = []
        for file in tqdm(file_list):
            feat = feature_extractor(file)
            features.append(feat)
        
        all_features = np.concatenate(features, axis=0)  # (total_frames, feature_dim)
        return all_features

    def get_precision(self, feature_type: str):
        assert feature_type in self.all_feature_type
        synth_wav_list = list()
        ref_wav_list = list()
        for speaker_id, wav_dict_list in self.speakers_to_synth_wavs_and_reference.items():
            for wav_dict in wav_dict_list:
                synth_wav_list.append(wav_dict["synthesized_wav"])
                ref_wav_list.append(wav_dict["reference_wav"])

        synth_features = self.get_feature(file_list=synth_wav_list, feature_type=feature_type)
        ref_features = self.get_feature(file_list=ref_wav_list, feature_type=feature_type)

        precision_value = self.manifold_estimate_faiss(
            A_features=synth_features,
            B_features=ref_features,
            k=self.k
        )
        return precision_value
    
    def __call__(self, feature_type):
        return self.get_precision(feature_type=feature_type)