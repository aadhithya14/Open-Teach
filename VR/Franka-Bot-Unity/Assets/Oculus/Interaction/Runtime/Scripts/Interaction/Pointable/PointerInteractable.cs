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
using UnityEngine.Assertions;

namespace Oculus.Interaction
{
    public abstract class PointerInteractable<TInteractor, TInteractable> : Interactable<TInteractor, TInteractable>,
        IPointable
        where TInteractor : Interactor<TInteractor, TInteractable>
        where TInteractable : PointerInteractable<TInteractor, TInteractable>
    {
        [SerializeField, Interface(typeof(IPointableElement)), Optional]
        private MonoBehaviour _pointableElement;

        public IPointableElement PointableElement { get; private set; }

        public event Action<PointerArgs> WhenPointerEventRaised = delegate { };

        protected bool _started = false;

        public void PublishPointerEvent(PointerArgs args)
        {
            WhenPointerEventRaised(args);
        }

        protected virtual void Awake()
        {
            if (_pointableElement != null)
            {
                PointableElement = _pointableElement as IPointableElement;
            }
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            if (_pointableElement != null)
            {
                Assert.IsNotNull(PointableElement);
            }
            this.EndStart(ref _started);
        }

        protected override void OnEnable()
        {
            base.OnEnable();
            if (_started)
            {
                if (PointableElement != null)
                {
                    WhenPointerEventRaised += PointableElement.ProcessPointerEvent;
                }
            }
        }

        protected override void OnDisable()
        {
            if (_started)
            {
                if (PointableElement != null)
                {
                    WhenPointerEventRaised -= PointableElement.ProcessPointerEvent;
                }
            }
            base.OnDisable();
        }

        #region Inject

        public void InjectOptionalPointableElement(IPointableElement pointableElement)
        {
            PointableElement = pointableElement;
            _pointableElement = pointableElement as MonoBehaviour;
        }

        #endregion
    }
}
