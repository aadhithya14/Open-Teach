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
    /// <summary>
    /// This component makes it possible to connect Interactors in the
    /// inspector to Unity Events that broadcast on state changes
    /// </summary>
    public class InteractorUnityEventWrapper : MonoBehaviour
    {
        [SerializeField, Interface(typeof(IInteractorView))]
        private MonoBehaviour _interactorView;
        private IInteractorView InteractorView;

        [SerializeField]
        private UnityEvent _whenHover;
        [SerializeField]
        private UnityEvent _whenUnhover;
        [SerializeField]
        private UnityEvent _whenSelect;
        [SerializeField]
        private UnityEvent _whenUnselect;

        public UnityEvent WhenHover => _whenHover;
        public UnityEvent WhenUnhover => _whenUnhover;
        public UnityEvent WhenSelect => _whenSelect;
        public UnityEvent WhenUnselect => _whenUnselect;

        protected bool _started = false;

        protected virtual void Awake()
        {
            InteractorView = _interactorView as IInteractorView;
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            Assert.IsNotNull(InteractorView);
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                InteractorView.WhenStateChanged += HandleStateChanged;
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                InteractorView.WhenStateChanged -= HandleStateChanged;
            }
        }

        private void HandleStateChanged(InteractorStateChangeArgs args)
        {
            switch (args.NewState)
            {
                case InteractorState.Normal:
                    if (args.PreviousState == InteractorState.Hover)
                    {
                        _whenUnhover.Invoke();
                    }

                    break;
                case InteractorState.Hover:
                    if (args.PreviousState == InteractorState.Normal)
                    {
                        _whenHover.Invoke();
                    }
                    else if (args.PreviousState == InteractorState.Select)
                    {
                        _whenUnselect.Invoke();
                    }

                    break;
                case InteractorState.Select:
                    if (args.PreviousState == InteractorState.Hover)
                    {
                        _whenSelect.Invoke();
                    }

                    break;
            }
        }

        #region Inject

        public void InjectAllInteractorUnityEventWrapper(IInteractorView interactorView)
        {
            InjectInteractorView(interactorView);
        }

        public void InjectInteractorView(IInteractorView interactorView)
        {
            _interactorView = interactorView as MonoBehaviour;
            InteractorView = interactorView;
        }

        #endregion
    }
}
