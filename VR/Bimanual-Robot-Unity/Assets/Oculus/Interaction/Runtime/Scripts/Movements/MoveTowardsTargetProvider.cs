/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;

namespace Oculus.Interaction.HandGrab
{
    public class MoveTowardsTargetProvider : MonoBehaviour, IMovementProvider
    {
        [SerializeField]
        private PoseTravelData _travellingData = PoseTravelData.FAST;

        public IMovement CreateMovement()
        {
            return new MoveTowardsTarget(_travellingData);
        }

        #region Inject
        public void InjectAllMoveTowardsTargetProvider(PoseTravelData travellingData)
        {
            InjectTravellingData(travellingData);
        }

        public void InjectTravellingData(PoseTravelData travellingData)
        {
            _travellingData = travellingData;
        }
        #endregion
    }

    public class MoveTowardsTarget : IMovement
    {
        private PoseTravelData _travellingData;

        public Pose Pose => _tween.Pose;
        public bool Stopped => _tween != null && _tween.Stopped;

        private Tween _tween;
        private Pose _source;
        private Pose _target;

        public MoveTowardsTarget(PoseTravelData travellingData)
        {
            _travellingData = travellingData;
        }

        public void MoveTo(Pose target)
        {
            _target = target;
            _tween = _travellingData.CreateTween(_source, target);
        }

        public void UpdateTarget(Pose target)
        {
            if (_target != target)
            {
                _target = target;
                _tween.UpdateTarget(_target);
            }
        }

        public void StopAndSetPose(Pose pose)
        {
            _source = pose;
            _tween?.StopAndSetPose(_source);
        }

        public void Tick()
        {
            _tween.Tick();
        }
    }
}
