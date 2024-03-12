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
    public class MoveFromTargetProvider : MonoBehaviour, IMovementProvider
    {
        public IMovement CreateMovement()
        {
            return new MoveFromTarget();
        }
    }

    public class MoveFromTarget : IMovement
    {
        public Pose Pose { get; private set; } = Pose.identity;
        public bool Stopped => true;

        public void StopMovement()
        {
        }

        public void MoveTo(Pose target)
        {
            Pose = target;
        }

        public void UpdateTarget(Pose target)
        {
            Pose = target;
        }

        public void StopAndSetPose(Pose source)
        {
            Pose = source;
        }

        public void Tick()
        {
        }
    }
}
