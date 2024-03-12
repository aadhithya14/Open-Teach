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
using UnityEngine.Assertions;
using UnityEngine.Events;
using UnityEngine.Serialization;

namespace Oculus.Interaction
{
    public class ActiveStateUnityEventWrapper : MonoBehaviour
    {
        [SerializeField, Interface(typeof(IActiveState))]
        private MonoBehaviour _activeState;
        private IActiveState ActiveState;

        [SerializeField]
        private UnityEvent _whenActivated;
        [SerializeField]
        private UnityEvent _whenDeactivated;

        public UnityEvent WhenActivated => _whenActivated;
        public UnityEvent WhenDeactivated => _whenDeactivated;

        [SerializeField]
        [Tooltip("If true, the corresponding event will be fired at the beginning of Update")]
        private bool _emitOnFirstUpdate = true;

        private bool _emittedOnFirstUpdate = false;

        private bool _savedState;

        protected virtual void Awake()
        {
            ActiveState = _activeState as IActiveState;
        }

        protected virtual void Start()
        {
            Assert.IsNotNull(ActiveState);
            _savedState = false;
        }

        protected virtual void Update()
        {
            if (_emitOnFirstUpdate && !_emittedOnFirstUpdate)
            {
                InvokeEvent();
                _emittedOnFirstUpdate = true;
            }

            if (_savedState != ActiveState.Active)
            {
                _savedState = ActiveState.Active;
                InvokeEvent();
            }
        }

        private void InvokeEvent()
        {
            if (_savedState)
            {
                _whenActivated.Invoke();
            }
            else
            {
                _whenDeactivated.Invoke();
            }
        }

        #region Inject

        public void InjectAllActiveStateUnityEventWrapper(IActiveState activeState)
        {
            InjectActiveState(activeState);
        }

        public void InjectActiveState(IActiveState activeState)
        {
            _activeState = activeState as MonoBehaviour;
            ActiveState = activeState;
        }

        public void InjectOptionalEmitOnFirstUpdate(bool emitOnFirstUpdate)
        {
            _emitOnFirstUpdate = emitOnFirstUpdate;
        }

        #endregion
    }
}
