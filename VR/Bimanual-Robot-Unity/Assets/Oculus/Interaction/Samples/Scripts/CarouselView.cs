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
using UnityEngine.UI;

namespace Oculus.Interaction.Samples
{
    public class CarouselView : MonoBehaviour
    {
        [SerializeField]
        private RectTransform _viewport;

        [SerializeField]
        private RectTransform _content;

        [SerializeField]
        private AnimationCurve _easeCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);

        [SerializeField, Optional]
        private GameObject _emptyCarouselVisuals;

        public int CurrentChildIndex => _currentChildIndex;

        public RectTransform ContentArea => _content;

        private int _currentChildIndex = 0;
        private float _scrollVal = 0;

        protected virtual void Start()
        {
            Assert.IsNotNull(_viewport);
            Assert.IsNotNull(_content);
        }

        public void ScrollRight()
        {
            if (_content.childCount <= 1)
            {
                return;
            }
            else if (_currentChildIndex > 0)
            {
                RectTransform currentChild = GetCurrentChild();
                _content.GetChild(0).SetAsLastSibling();
                LayoutRebuilder.ForceRebuildLayoutImmediate(_content);
                ScrollToChild(currentChild, 1);
            }
            else
            {
                _currentChildIndex++;
            }
            _scrollVal = Time.time;
        }

        public void ScrollLeft()
        {
            if (_content.childCount <= 1)
            {
                return;
            }
            else if (_currentChildIndex < _content.childCount - 1)
            {
                RectTransform currentChild = GetCurrentChild();
                _content.GetChild(_content.childCount - 1).SetAsFirstSibling();
                LayoutRebuilder.ForceRebuildLayoutImmediate(_content);
                ScrollToChild(currentChild, 1);
            }
            else
            {
                _currentChildIndex--;
            }
            _scrollVal = Time.time;
        }

        private RectTransform GetCurrentChild()
        {
            return _content.GetChild(_currentChildIndex) as RectTransform;
        }

        private void ScrollToChild(RectTransform child, float amount01)
        {
            if (child == null)
            {
                return;
            }

            amount01 = Mathf.Clamp01(amount01);

            Vector3 viewportCenter = _viewport.TransformPoint(_viewport.rect.center);
            Vector3 imageCenter = child.TransformPoint(child.rect.center);
            Vector3 offset = imageCenter - viewportCenter;

            if (offset.sqrMagnitude > float.Epsilon)
            {
                Vector3 targetPosition = _content.position - offset;
                float lerp = Mathf.Clamp01(_easeCurve.Evaluate(amount01));
                _content.position = Vector3.Lerp(_content.position, targetPosition, lerp);
            }
        }

        protected virtual void Update()
        {
            _currentChildIndex = Mathf.Clamp(
                _currentChildIndex, 0, _content.childCount - 1);

            bool hasImages = _content.childCount > 0;
            if (hasImages)
            {
                RectTransform currentImage = GetCurrentChild();
                ScrollToChild(currentImage, Time.time - _scrollVal);
            }

            if (_emptyCarouselVisuals != null)
            {
                _emptyCarouselVisuals.SetActive(!hasImages);
            }
        }
    }
}
