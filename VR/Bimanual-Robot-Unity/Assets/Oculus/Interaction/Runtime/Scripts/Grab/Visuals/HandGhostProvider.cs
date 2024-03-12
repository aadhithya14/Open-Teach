/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.Input;
using UnityEngine;

namespace Oculus.Interaction.HandGrab.Visuals
{
    /// <summary>
    /// Holds references to the prefabs for Ghost-Hands, so they can be instantiated
    /// in runtime to represent static poses.
    /// </summary>
    [CreateAssetMenu(menuName = "Oculus/Interaction/SDK/Pose Authoring/Hand Ghost Provider")]
    public class HandGhostProvider : ScriptableObject
    {
        /// <summary>
        /// The prefab for the left hand ghost.
        /// </summary>
        [SerializeField]
        private HandGhost _leftHand;
        /// <summary>
        /// The prefab for the right hand ghost.
        /// </summary>
        [SerializeField]
        private HandGhost _rightHand;

        /// <summary>
        /// Helper method to obtain the prototypes
        /// The result is to be instanced, not used directly.
        /// </summary>
        /// <param name="handedness">The desired handedness of the ghost prefab</param>
        /// <returns>A Ghost prefab</returns>
        public HandGhost GetHand(Handedness handedness)
        {
            return handedness == Handedness.Left ? _leftHand : _rightHand;
        }
    }
}
