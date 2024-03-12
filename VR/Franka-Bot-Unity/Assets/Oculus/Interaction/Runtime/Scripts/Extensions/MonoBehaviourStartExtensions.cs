/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using UnityEngine;

namespace Oculus.Interaction
{
    /// <summary>
    /// By adding BeginStart and EndStart at the beginning and end of Start, MonoBehaviours with
    /// OnEnable and OnDisable logic can wrap their contents within a _started flag and effectively
    /// skip over logic in those methods until after Start has been invoked.
    ///
    /// To not bypass the Unity Lifecycle, the enabled property is used to disable the most derived
    /// MonoBehaviour, invoke Start up the hierarchy chain, and finally re-enable the MonoBehaviour.
    /// </summary>
    public static class MonoBehaviourStartExtensions
    {
        public static void BeginStart(this MonoBehaviour monoBehaviour, ref bool started,
                                      Action baseStart = null)
        {
            if (!started)
            {
                monoBehaviour.enabled = false;
                started = true;
                baseStart?.Invoke();
                started = false;
            }
            else
            {
                baseStart?.Invoke();
            }
        }

        public static void EndStart(this MonoBehaviour monoBehaviour, ref bool started)
        {
            if (!started)
            {
                started = true;
                monoBehaviour.enabled = true;
            }
        }
    }
}
