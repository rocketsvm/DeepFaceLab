import math
import multiprocessing
import time
import traceback

import cv2
import numpy as np

from core import mplib
from core.interact import interact as io
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)


'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFace(SampleGeneratorBase):
    def __init__ (self, samples_path, debug=False, batch_size=1,
                        random_ct_samples_path=None,
                        sample_process_options=SampleProcessor.Options(),
                        output_sample_types=[],
                        uniform_yaw_distribution=False,
                        uniform_pitch_distribution=False,
                        generators_count=4,
                        raise_on_no_data=True,                        
                        **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        
        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        samples = SampleLoader.load (SampleType.FACE, samples_path)
        self.samples_len = len(samples)
        
        if self.samples_len == 0:
            if raise_on_no_data:
                raise ValueError('No training data provided.')
            else:
                return
                
        if uniform_yaw_distribution or uniform_pitch_distribution:
            samples_pyr = [ ( idx, sample.get_pitch_yaw_roll() ) for idx, sample in enumerate(samples) ]
            
            grads = 128
            yaws_sample_list = [None]*grads
            pitches_sample_list = [None]*grads

            if uniform_yaw_distribution:
                #instead of math.pi / 2, using -1.2,+1.2 because actually maximum yaw for 2DFAN landmarks are -1.2+1.2
                grads_space = np.linspace (-1.2, 1.2,grads)
                for g in io.progress_bar_generator ( range(grads), "Sort by yaw"):
                    yaw = grads_space[g]
                    next_yaw = grads_space[g+1] if g < grads-1 else yaw

                    yaw_samples = []
                    for idx, pyr in samples_pyr:
                        s_yaw = -pyr[1]
                        if (g == 0          and s_yaw < next_yaw) or \
                        (g < grads-1     and s_yaw >= yaw and s_yaw < next_yaw) or \
                        (g == grads-1    and s_yaw >= yaw):
                            yaw_samples += [ idx ]
                    if len(yaw_samples) > 0:
                        yaws_sample_list[g] = yaw_samples
                
            if uniform_pitch_distribution:
                grads_space = np.linspace (-math.pi / 2,math.pi / 2, grads )
                for g in io.progress_bar_generator ( range(grads), "Sort by pitch"):
                    pitch = grads_space[g]
                    next_pitch = grads_space[g+1] if g < grads-1 else pitch

                    pitch_samples = []
                    for idx, pyr in samples_pyr:
                        s_pitch = -pyr[0]
                        if (g == 0          and s_pitch < next_pitch) or \
                        (g < grads-1     and s_pitch >= pitch and s_pitch < next_pitch) or \
                        (g == grads-1    and s_pitch >= pitch):
                            pitch_samples += [ idx ]
                    if len(pitch_samples) > 0:
                        pitches_sample_list[g] = pitch_samples  # TODO: do not include duplicates from yaws_sample_list

            selected_sample_list = yaws_sample_list + pitches_sample_list
            selected_sample_list = [ s for s in selected_sample_list if s is not None ]
            index_host = mplib.Index2DHost( selected_sample_list )
            
        else:
            index_host = mplib.IndexHost(self.samples_len)

        if random_ct_samples_path is not None:
            ct_samples = SampleLoader.load (SampleType.FACE, random_ct_samples_path)
            ct_index_host = mplib.IndexHost( len(ct_samples) )
        else:
            ct_samples = None
            ct_index_host = None

        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (samples, index_host.create_cli(), ct_samples, ct_index_host.create_cli() if ct_index_host is not None else None) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (samples, index_host.create_cli(), ct_samples, ct_index_host.create_cli() if ct_index_host is not None else None), start_now=False ) \
                               for i in range(self.generators_count) ]
                               
            SubprocessGenerator.start_in_parallel( self.generators )

        self.generator_counter = -1
        
        self.initialized = True
        
    #overridable
    def is_initialized(self):
        return self.initialized
        
    def __iter__(self):
        return self

    def __next__(self):
        if not self.initialized:
            return []
            
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        samples, index_host, ct_samples, ct_index_host = param
 
        bs = self.batch_size
        while True:
            batches = None

            indexes = index_host.multi_get(bs)
            ct_indexes = ct_index_host.multi_get(bs) if ct_samples is not None else None

            t = time.time()
            for n_batch in range(bs):
                sample_idx = indexes[n_batch]
                sample = samples[sample_idx]

                ct_sample = None
                if ct_samples is not None:
                    ct_sample = ct_samples[ct_indexes[n_batch]]

                try:
                    x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug, ct_sample=ct_sample)
                except:
                    raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                if batches is None:
                    batches = [ [] for _ in range(len(x)) ]

                for i in range(len(x)):
                    batches[i].append ( x[i] )

            yield [ np.array(batch) for batch in batches]
