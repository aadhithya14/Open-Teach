/**************************************************************************************************
 * Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.
 *
 * Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 **************************************************************************************************/

using System;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace Oculus.Voice.Demo
{
    public class ButtonEventWatcher : MonoBehaviour
    {
#if ENABLE_LEGACY_INPUT_MANAGER
        // By default: uses space bar, oculus quest a button and oculus quest x button
        [SerializeField] private KeyCode[] _keys = new KeyCode[] { KeyCode.Space, KeyCode.JoystickButton0, KeyCode.JoystickButton2 };
#endif

        // Used for click or hold events
        public UnityEvent OnButtonDown;
        // Used for button up hold events
        public UnityEvent OnButtonUp;

#if ENABLE_LEGACY_INPUT_MANAGER
        // Update activation
        void Update()
        {
            // Iterate keys
            foreach (var key in _keys)
            {
                if (Input.GetKeyDown(key))
                {
                    OnButtonDown?.Invoke();
                }
                else if (Input.GetKeyUp(key))
                {
                    OnButtonUp?.Invoke();
                }
            }
        }
#endif
    }
}
