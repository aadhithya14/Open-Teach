from openteach.components import Component

class Recorder(Component):
    def _add_metadata(self, datapoints):
        self.metadata = dict(
            file_name = self._recorder_file_name,
            num_datapoints = datapoints,
            record_start_time = self.record_start_time,
            record_end_time = self.record_end_time,
            record_duration = self.record_end_time - self.record_start_time,
            record_frequency = datapoints / (self.record_end_time - self.record_start_time)
        )

    def _display_statistics(self, datapoints):
        print('Saving data to {}'.format(self._recorder_file_name))
        print('Number of datapoints recorded: {}.'.format(datapoints))
        print('Data record frequency: {}.'.format(datapoints / (self.record_end_time - self.record_start_time)))
