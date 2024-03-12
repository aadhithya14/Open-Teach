/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

namespace Oculus.Interaction.Input
{
    public class LastKnownGoodHand : Hand
    {
        private readonly HandDataAsset _lastState = new HandDataAsset();

        protected override void Apply(HandDataAsset data)
        {
            bool shouldUseData = data.IsHighConfidence ||
                                 data.RootPoseOrigin == PoseOrigin.FilteredTrackedPose ||
                                 data.RootPoseOrigin == PoseOrigin.SyntheticPose;
            if (data.IsDataValid && data.IsTracked && shouldUseData)
            {
                _lastState.CopyFrom(data);
            }
            else if (_lastState.IsDataValid && data.IsConnected)
            {
                // No high confidence data, use last known good.
                // Only copy pose data, not confidence/tracked flags.
                data.CopyPosesFrom(_lastState);
                data.RootPoseOrigin = PoseOrigin.SyntheticPose;
                data.IsDataValid = true;
                data.IsTracked = true;
                data.IsHighConfidence = true;
            }
            else
            {
                // This hand is not connected, or has never seen valid data.
                data.IsTracked = false;
                data.IsHighConfidence = false;
                data.RootPoseOrigin = PoseOrigin.None;
            }
        }
    }
}
